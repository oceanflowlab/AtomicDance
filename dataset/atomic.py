"""Utilities for atomic movement plans and prototype retrieval."""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class AtomicSegment:
    label: int
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def labels_to_segments(labels: torch.Tensor) -> List[AtomicSegment]:
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if labels.numel() == 0:
        return []
    changes = torch.nonzero(labels[1:] != labels[:-1], as_tuple=False).flatten() + 1
    bounds = [0, *changes.tolist(), labels.numel()]
    return [
        AtomicSegment(int(labels[start].item()), start, end)
        for start, end in zip(bounds[:-1], bounds[1:])
    ]


def majority_vote(labels: torch.Tensor, window_size: int = 5) -> torch.Tensor:
    """Centered sliding-window vote with deterministic center-label tie breaking."""
    if labels.ndim not in (1, 2):
        raise ValueError("labels must have shape [frames] or [batch, frames]")
    if window_size < 1 or window_size % 2 == 0:
        raise ValueError("window_size must be a positive odd number")
    batched = labels.unsqueeze(0) if labels.ndim == 1 else labels
    if batched.shape[1] == 0:
        return labels.clone()
    radius = window_size // 2
    if radius:
        left = batched[:, :1].expand(-1, radius)
        right = batched[:, -1:].expand(-1, radius)
        padded = torch.cat((left, batched, right), dim=1)
    else:
        padded = batched
    windows = padded.unfold(1, window_size, 1)
    num_classes = int(batched.max().item()) + 1 if batched.numel() else 1
    counts = F.one_hot(windows, num_classes=num_classes).sum(dim=-2)
    winners = counts.argmax(dim=-1)
    center = batched
    max_counts = counts.max(dim=-1).values
    center_counts = counts.gather(-1, center.unsqueeze(-1)).squeeze(-1)
    winners = torch.where(center_counts == max_counts, center, winners)
    return winners.squeeze(0) if labels.ndim == 1 else winners


def merge_short_segments(
    labels: torch.Tensor,
    min_length: int,
    compatibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Merge short runs into a neighboring run until every run is long enough.

    If a class-compatibility matrix is provided, its [source, target] value is
    used first. Ties are resolved in favor of the longer neighbor.
    """
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if min_length <= 1:
        return labels.clone()
    result = labels.clone()
    while True:
        segments = labels_to_segments(result)
        short_index = next((i for i, segment in enumerate(segments) if segment.length < min_length), None)
        if short_index is None or len(segments) == 1:
            break
        segment = segments[short_index]
        candidates = []
        for index in (short_index - 1, short_index + 1):
            if 0 <= index < len(segments):
                neighbor = segments[index]
                score = float(neighbor.length)
                if compatibility is not None:
                    score += float(compatibility[segment.label, neighbor.label]) * 1e6
                candidates.append((score, neighbor.label))
        target = max(candidates, key=lambda item: item[0])[1]
        result[segment.start : segment.end] = target
    return result


def refine_plan(
    labels: torch.Tensor,
    vote_window: int = 5,
    min_length: int = 6,
    compatibility: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    voted = majority_vote(labels, vote_window)
    if voted.ndim == 1:
        return merge_short_segments(voted, min_length, compatibility)
    return torch.stack(
        [merge_short_segments(row, min_length, compatibility) for row in voted]
    )


def plan_boundaries(labels: torch.Tensor) -> torch.Tensor:
    """Mark the first frame of every new atomic/transition segment."""
    if labels.ndim not in (1, 2):
        raise ValueError("labels must have shape [frames] or [batch, frames]")
    batched = labels.unsqueeze(0) if labels.ndim == 1 else labels
    boundaries = torch.zeros_like(batched, dtype=torch.bool)
    boundaries[:, 1:] = batched[:, 1:] != batched[:, :-1]
    return boundaries.squeeze(0) if labels.ndim == 1 else boundaries


class AtomicMotionLibrary:
    """In-memory prototypes indexed by atomic label, as described in Sec. 3.3."""

    def __init__(self, motions: Mapping[int, Sequence[torch.Tensor]]) -> None:
        self.motions: Dict[int, Tuple[torch.Tensor, ...]] = {
            int(label): tuple(segment.detach().clone() for segment in segments)
            for label, segments in motions.items()
        }
        if 0 in self.motions:
            raise ValueError("label 0 is reserved for transitions")

    @classmethod
    def from_sequences(
        cls,
        motions: Sequence[torch.Tensor],
        labels: Sequence[torch.Tensor],
        min_length: int = 1,
    ) -> "AtomicMotionLibrary":
        if len(motions) != len(labels):
            raise ValueError("motions and labels must contain the same number of sequences")
        groups: Dict[int, List[torch.Tensor]] = {}
        for motion, plan in zip(motions, labels):
            if motion.shape[0] != plan.shape[0]:
                raise ValueError("motion and label frame counts must match")
            for segment in labels_to_segments(plan):
                if segment.label and segment.length >= min_length:
                    groups.setdefault(segment.label, []).append(motion[segment.start : segment.end])
        return cls(groups)

    def state_dict(self):
        return {label: [motion.clone() for motion in motions] for label, motions in self.motions.items()}

    @classmethod
    def from_state_dict(cls, state):
        return cls(state)

    def retrieve(self, label: int, target_length: int) -> torch.Tensor:
        if label == 0:
            raise ValueError("transition frames do not have atomic prototypes")
        candidates = self.motions.get(int(label), ())
        if not candidates:
            raise KeyError(f"no prototype candidates for atomic label {label}")
        chosen = min(candidates, key=lambda motion: abs(motion.shape[0] - target_length))
        return self._resample(chosen, target_length)

    @staticmethod
    def _resample(motion: torch.Tensor, target_length: int) -> torch.Tensor:
        if motion.ndim != 2:
            raise ValueError("motion must have shape [frames, features]")
        if target_length < 1:
            raise ValueError("target_length must be positive")
        if motion.shape[0] == target_length:
            return motion.clone()
        values = motion.transpose(0, 1).unsqueeze(0)
        return F.interpolate(values, size=target_length, mode="linear", align_corners=True).squeeze(0).transpose(0, 1)

    def build_draft(self, labels: torch.Tensor, feature_dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return coarse motion M0 and mask w; transition frames remain zero."""
        draft = torch.zeros(labels.shape[0], feature_dim, dtype=torch.float32, device=labels.device)
        mask = torch.zeros(labels.shape[0], 1, dtype=torch.float32, device=labels.device)
        for segment in labels_to_segments(labels):
            if segment.label == 0:
                continue
            motion = self.retrieve(segment.label, segment.length).to(labels.device)
            if motion.shape[1] != feature_dim:
                raise ValueError("prototype feature dimension does not match requested draft")
            draft[segment.start : segment.end] = motion
            mask[segment.start : segment.end] = 1.0
        return draft, mask
