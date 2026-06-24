"""Transition-aware continuous diffusion for atomic dance completion."""

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import DanceDecoder
from model.utils import extract, make_beta_schedule


class AtomicCompletionDecoder(nn.Module):
    """EDGE denoiser augmented with retrieved motion draft M0 and noise mask w."""

    def __init__(
        self,
        motion_dim: int,
        seq_len: int,
        music_dim: int,
        latent_dim: int = 512,
        ff_size: int = 1024,
        num_layers: int = 8,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.motion_dim = motion_dim
        self.input_adapter = nn.Sequential(
            nn.Linear(motion_dim * 2 + 1, motion_dim),
            nn.SiLU(),
            nn.Linear(motion_dim, motion_dim),
        )
        self.denoiser = DanceDecoder(
            nfeats=motion_dim,
            seq_len=seq_len,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            cond_feature_dim=music_dim,
            activation=F.gelu,
        )

    def forward(
        self,
        noisy_motion: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        cond_drop_prob: float = 0.0,
    ) -> torch.Tensor:
        if noise_mask.shape[-1] != 1:
            raise ValueError("noise_mask must have shape [batch, frames, 1]")
        fused = self.input_adapter(torch.cat((noisy_motion, draft_motion, noise_mask), dim=-1))
        return self.denoiser(fused, music_features, timesteps, cond_drop_prob=cond_drop_prob)

    def guided_forward(
        self,
        noisy_motion: torch.Tensor,
        music_features: torch.Tensor,
        timesteps: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        guidance_weight: float,
    ) -> torch.Tensor:
        unconditional = self.forward(
            noisy_motion, music_features, timesteps, draft_motion, noise_mask, cond_drop_prob=1.0
        )
        conditional = self.forward(
            noisy_motion, music_features, timesteps, draft_motion, noise_mask, cond_drop_prob=0.0
        )
        return unconditional + guidance_weight * (conditional - unconditional)


@dataclass
class CompletionLoss:
    total: torch.Tensor
    denoising: torch.Tensor
    transition: torch.Tensor
    prediction: torch.Tensor


class AtomicCompletionDiffusion(nn.Module):
    """Clean-motion-predicting DDPM corresponding to Eq. (4) in the paper."""

    def __init__(
        self,
        model: AtomicCompletionDecoder,
        num_steps: int = 1000,
        schedule: str = "cosine",
        transition_weight: float = 1.0,
        cond_drop_prob: float = 0.25,
        guidance_weight: float = 2.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.num_steps = num_steps
        self.transition_weight = transition_weight
        self.cond_drop_prob = cond_drop_prob
        self.guidance_weight = guidance_weight

        betas = torch.as_tensor(make_beta_schedule(schedule, num_steps), dtype=torch.float32)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())
        self.register_buffer(
            "posterior_variance",
            betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * alpha_bars_previous.sqrt() / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alpha_bars_previous) * alphas.sqrt() / (1.0 - alpha_bars),
        )

    def q_sample(
        self, clean_motion: torch.Tensor, timesteps: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        noise = torch.randn_like(clean_motion) if noise is None else noise
        return (
            extract(self.sqrt_alpha_bars, timesteps, clean_motion.shape) * clean_motion
            + extract(self.sqrt_one_minus_alpha_bars, timesteps, clean_motion.shape) * noise
        )

    @staticmethod
    def perturb_draft(draft_motion: torch.Tensor, noise_mask: torch.Tensor) -> torch.Tensor:
        """Apply appropriately scaled noise only to retrieved atomic frames."""
        return draft_motion + torch.randn_like(draft_motion) * noise_mask

    @staticmethod
    def transition_loss(prediction: torch.Tensor, boundaries: Optional[torch.Tensor]) -> torch.Tensor:
        if boundaries is None:
            return prediction.new_zeros(())
        if boundaries.shape != prediction.shape[:2]:
            raise ValueError("boundaries must have shape [batch, frames]")
        velocity = (prediction[:, 1:] - prediction[:, :-1]).abs().mean(dim=-1)
        selected = boundaries[:, 1:].to(dtype=velocity.dtype)
        return (velocity * selected).sum() / selected.sum().clamp_min(1.0)

    def training_step(
        self,
        clean_motion: torch.Tensor,
        music_features: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        boundaries: Optional[torch.Tensor] = None,
        timesteps: Optional[torch.Tensor] = None,
    ) -> CompletionLoss:
        batch = clean_motion.shape[0]
        if timesteps is None:
            timesteps = torch.randint(self.num_steps, (batch,), device=clean_motion.device)
        noisy_motion = self.q_sample(clean_motion, timesteps)
        noisy_draft = self.perturb_draft(draft_motion, noise_mask)
        prediction = self.model(
            noisy_motion,
            music_features,
            timesteps,
            noisy_draft,
            noise_mask,
            cond_drop_prob=self.cond_drop_prob,
        )
        denoising = F.mse_loss(prediction, clean_motion)
        transition = self.transition_loss(prediction, boundaries)
        return CompletionLoss(
            denoising + self.transition_weight * transition,
            denoising,
            transition,
            prediction,
        )

    @torch.no_grad()
    def sample(
        self,
        music_features: torch.Tensor,
        draft_motion: torch.Tensor,
        noise_mask: torch.Tensor,
        guidance_weight: Optional[float] = None,
    ) -> torch.Tensor:
        weight = self.guidance_weight if guidance_weight is None else guidance_weight
        motion = torch.randn_like(draft_motion)
        conditioned_draft = self.perturb_draft(draft_motion, noise_mask)
        for step in reversed(range(self.num_steps)):
            timesteps = torch.full(
                (motion.shape[0],), step, dtype=torch.long, device=motion.device
            )
            clean = self.model.guided_forward(
                motion,
                music_features,
                timesteps,
                conditioned_draft,
                noise_mask,
                weight,
            ).clamp(-1.0, 1.0)
            mean = (
                extract(self.posterior_mean_coef1, timesteps, motion.shape) * clean
                + extract(self.posterior_mean_coef2, timesteps, motion.shape) * motion
            )
            if step > 0:
                variance = extract(self.posterior_variance, timesteps, motion.shape)
                motion = mean + variance.sqrt() * torch.randn_like(motion)
            else:
                motion = mean
        return motion
