"""Paper-specified visual-similarity segmentation and base clustering."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F


def kmeans(
    features: torch.Tensor,
    num_clusters: int,
    iterations: int = 50,
    seed: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Small dependency-free K-Means used for reproducible preprocessing."""
    if features.ndim != 2 or not 0 < num_clusters <= features.shape[0]:
        raise ValueError("invalid features or number of clusters")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    centers = features[torch.randperm(features.shape[0], generator=generator, device=features.device)[:num_clusters]].clone()
    labels = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
    for _ in range(iterations):
        new_labels = torch.cdist(features, centers).argmin(dim=1)
        new_centers = []
        for cluster in range(num_clusters):
            members = features[new_labels == cluster]
            new_centers.append(members.mean(dim=0) if len(members) else centers[cluster])
        new_centers = torch.stack(new_centers)
        if torch.equal(new_labels, labels) and torch.allclose(new_centers, centers):
            labels, centers = new_labels, new_centers
            break
        labels, centers = new_labels, new_centers
    return labels, centers


def _runs(labels: torch.Tensor) -> List[Tuple[int, int, int]]:
    changes = torch.nonzero(labels[1:] != labels[:-1], as_tuple=False).flatten() + 1
    boundaries = [0, *changes.tolist(), labels.numel()]
    return [(start, end, int(labels[start])) for start, end in zip(boundaries[:-1], boundaries[1:])]


def adaptive_segment(
    visual_features: torch.Tensor,
    motion: torch.Tensor,
    num_clusters: int,
    min_length: int,
    iterations: int = 50,
    seed: int = 0,
) -> Tuple[List[torch.Tensor], List[int]]:
    """Algorithm 1: cluster augmented rows of the visual similarity matrix."""
    if visual_features.shape[0] != motion.shape[0]:
        raise ValueError("visual features and motion must have the same frame count")
    normalized = F.normalize(visual_features.float(), dim=-1)
    similarity = normalized @ normalized.transpose(0, 1)
    time = torch.linspace(0.0, 1.0, visual_features.shape[0], device=visual_features.device)[:, None]
    augmented = torch.cat((similarity, time), dim=-1)
    frame_labels, _ = kmeans(augmented, num_clusters, iterations=iterations, seed=seed)

    # Iteratively eliminate short runs. Algorithm 1 selects the shorter of the
    # two temporal neighbors; edge runs have only one possible neighbor.
    while True:
        runs = _runs(frame_labels)
        short_index = next((i for i, (start, end, _) in enumerate(runs) if end - start < min_length), None)
        if short_index is None or len(runs) == 1:
            break
        start, end, _ = runs[short_index]
        neighbors = []
        for index in (short_index - 1, short_index + 1):
            if 0 <= index < len(runs):
                n_start, n_end, n_label = runs[index]
                neighbors.append((n_end - n_start, n_label))
        frame_labels[start:end] = min(neighbors, key=lambda item: item[0])[1]

    runs = _runs(frame_labels)
    cut_points = [end for _, end, _ in runs[:-1]]
    segments = [motion[start:end].clone() for start, end, _ in runs]
    return segments, cut_points


@dataclass
class BaseClusterResult:
    labels: torch.Tensor
    centers: torch.Tensor
    keep_mask: torch.Tensor


def cluster_motion_embeddings(
    embeddings: torch.Tensor,
    num_clusters: int = 100,
    keep_quantile: float = 0.8,
    iterations: int = 100,
    seed: int = 0,
) -> BaseClusterResult:
    """Cluster TMR embeddings and discard ambiguous points far from centers."""
    if not 0.0 < keep_quantile <= 1.0:
        raise ValueError("keep_quantile must be in (0, 1]")
    labels, centers = kmeans(embeddings, num_clusters, iterations=iterations, seed=seed)
    distances = (embeddings - centers[labels]).norm(dim=-1)
    keep = torch.zeros_like(labels, dtype=torch.bool)
    for cluster in range(num_clusters):
        indices = torch.nonzero(labels == cluster, as_tuple=False).flatten()
        if not len(indices):
            continue
        count = max(1, int(round(len(indices) * keep_quantile)))
        chosen = indices[distances[indices].argsort()[:count]]
        keep[chosen] = True
    return BaseClusterResult(labels, centers, keep)
