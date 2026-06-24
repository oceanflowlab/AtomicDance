"""Discrete diffusion planner for frame-wise atomic movement labels.
"""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.utils import PositionalEncoding, SinusoidalPosEmb, make_beta_schedule


@dataclass
class PlannerOutput:
    loss: torch.Tensor
    logits: torch.Tensor
    noisy_labels: torch.Tensor
    target_labels: torch.Tensor
    timesteps: torch.Tensor


class AtomicPlannerTransformer(nn.Module):
    """Music-conditioned Transformer used as the D3PM reverse model."""

    def __init__(
        self,
        num_atomic_classes: int,
        music_dim: int,
        latent_dim: int = 512,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_size: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 18000,
    ) -> None:
        super().__init__()
        self.num_classes = num_atomic_classes + 1
        self.label_embedding = nn.Embedding(self.num_classes, latent_dim)
        self.music_projection = nn.Linear(music_dim, latent_dim)
        self.time_embedding = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, latent_dim * 4),
            nn.SiLU(),
            nn.Linear(latent_dim * 4, latent_dim),
        )
        self.position = PositionalEncoding(
            latent_dim, dropout=dropout, max_len=max_seq_len, batch_first=True
        )
        layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=ff_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output = nn.Sequential(nn.LayerNorm(latent_dim), nn.Linear(latent_dim, self.num_classes))

    def forward(
        self,
        noisy_labels: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if noisy_labels.ndim != 2:
            raise ValueError("noisy_labels must have shape [batch, frames]")
        if music_features.shape[:2] != noisy_labels.shape:
            raise ValueError("music_features and labels must share batch/frame dimensions")
        x = self.label_embedding(noisy_labels)
        x = x + self.music_projection(music_features)
        x = x + self.time_embedding(timesteps)[:, None, :]
        x = self.position(x)
        return self.output(self.encoder(x, src_key_padding_mask=padding_mask))


class UniformD3PM(nn.Module):
    """Uniform categorical diffusion with a directly parameterized reverse step."""

    def __init__(
        self,
        model: AtomicPlannerTransformer,
        num_steps: int = 100,
        schedule: str = "cosine",
    ) -> None:
        super().__init__()
        self.model = model
        self.num_classes = model.num_classes
        self.num_steps = num_steps
        betas = torch.as_tensor(make_beta_schedule(schedule, num_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", torch.cumprod(alphas, dim=0))

    def _uniform_replace(self, labels: torch.Tensor, keep_prob: torch.Tensor) -> torch.Tensor:
        while keep_prob.ndim < labels.ndim:
            keep_prob = keep_prob.unsqueeze(-1)
        keep = torch.rand(labels.shape, device=labels.device) < keep_prob
        random_labels = torch.randint(self.num_classes, labels.shape, device=labels.device)
        return torch.where(keep, labels, random_labels)

    def q_sample(self, clean_labels: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample q(y_t | y_0) using the cumulative token-retention rate."""
        keep_prob = self.alpha_bars.gather(0, timesteps)
        return self._uniform_replace(clean_labels, keep_prob)

    def q_step(self, labels: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Sample q(y_t | y_{t-1}); timesteps use zero-based diffusion indices."""
        keep_prob = self.alphas.gather(0, timesteps)
        return self._uniform_replace(labels, keep_prob)

    def training_step(
        self,
        clean_labels: torch.Tensor,
        music_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> PlannerOutput:
        batch = clean_labels.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.num_steps, (batch,), device=clean_labels.device)

        # Construct the exact adjacent pair used in Eq. (3): y_{t-1} -> y_t.
        previous_t = torch.clamp(timesteps - 1, min=0)
        previous = self.q_sample(clean_labels, previous_t)
        previous = torch.where((timesteps == 0)[:, None], clean_labels, previous)
        noisy = self.q_step(previous, timesteps)
        logits = self.model(noisy, music_features, timesteps, padding_mask)

        per_token = F.cross_entropy(logits.transpose(1, 2), previous, reduction="none")
        if padding_mask is not None:
            valid = ~padding_mask
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
        else:
            loss = per_token.mean()
        return PlannerOutput(loss, logits, noisy, previous, timesteps)

    @torch.no_grad()
    def sample(
        self,
        music_features: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
        temperature: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        labels = torch.randint(
            self.num_classes, music_features.shape[:2], device=music_features.device
        )
        for step in reversed(range(self.num_steps)):
            t = torch.full((labels.shape[0],), step, device=labels.device, dtype=torch.long)
            logits = self.model(labels, music_features, t, padding_mask) / temperature
            if deterministic:
                labels = logits.argmax(dim=-1)
            else:
                labels = torch.distributions.Categorical(logits=logits).sample()
        if padding_mask is not None:
            labels = labels.masked_fill(padding_mask, 0)
        return labels
