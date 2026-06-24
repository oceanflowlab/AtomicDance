"""Frame-aligned full-sequence dataset for the two-stage atomic model."""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class AtomicSequenceDataset(Dataset):
    """Load matching NumPy files from motion/music/labels directories.

    Expected layout::

        root/{train,test}/motion/<sequence>.npy
        root/{train,test}/music/<sequence>.npy
        root/{train,test}/labels/<sequence>.npy
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        window_size: Optional[int] = None,
        random_crop: bool = True,
    ) -> None:
        self.root = Path(root) / split
        self.window_size = window_size
        self.random_crop = random_crop
        self.arrays = None
        array_paths = [self.root / name for name in ("motion.npy", "music.npy", "labels.npy")]
        names_path = self.root / "names.json"
        if all(path.is_file() for path in array_paths) and names_path.is_file():
            import json

            self.arrays = tuple(np.load(str(path), mmap_mode="r") for path in array_paths)
            with open(str(names_path), "r") as handle:
                self.names = json.load(handle)
            if not (len(self.arrays[0]) == len(self.arrays[1]) == len(self.arrays[2]) == len(self.names)):
                raise ValueError(f"unaligned indexed arrays under {self.root}")
            self.samples = list(range(len(self.names)))
            return

        motion_dir = self.root / "motion"
        music_dir = self.root / "music"
        label_dir = self.root / "labels"
        self.samples = []
        for motion_path in sorted(motion_dir.glob("*.npy")):
            music_path = music_dir / motion_path.name
            label_path = label_dir / motion_path.name
            if not music_path.is_file() or not label_path.is_file():
                raise FileNotFoundError(f"missing aligned music/labels for {motion_path.stem}")
            self.samples.append((motion_path, music_path, label_path))
        if not self.samples:
            raise FileNotFoundError(f"no .npy sequences found under {motion_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        if self.arrays is not None:
            motion = torch.from_numpy(np.array(self.arrays[0][index], copy=True)).float()
            music = torch.from_numpy(np.array(self.arrays[1][index], copy=True)).float()
            labels = torch.from_numpy(np.array(self.arrays[2][index], copy=True)).long()
            name = self.names[index]
        else:
            motion_path, music_path, label_path = self.samples[index]
            motion = torch.from_numpy(np.load(motion_path)).float()
            music = torch.from_numpy(np.load(music_path)).float()
            labels = torch.from_numpy(np.load(label_path)).long()
            name = motion_path.stem
        if motion.ndim != 2 or music.ndim != 2 or labels.ndim != 1:
            raise ValueError(f"invalid tensor rank in sequence {name}")
        if not (motion.shape[0] == music.shape[0] == labels.shape[0]):
            raise ValueError(f"unaligned frame counts in sequence {name}")

        if self.window_size is not None:
            if motion.shape[0] < self.window_size:
                raise ValueError(
                    f"sequence {name} is shorter than window_size={self.window_size}"
                )
            maximum_start = motion.shape[0] - self.window_size
            if self.random_crop and maximum_start:
                start = int(torch.randint(maximum_start + 1, ()).item())
            else:
                start = maximum_start // 2
            selection = slice(start, start + self.window_size)
            motion, music, labels = motion[selection], music[selection], labels[selection]
        return {"motion": motion, "music": music, "labels": labels, "name": name}


def collate_atomic_sequences(samples):
    """Pad full songs for planner training and expose a Transformer padding mask."""
    lengths = torch.tensor([sample["labels"].shape[0] for sample in samples], dtype=torch.long)
    motion = pad_sequence([sample["motion"] for sample in samples], batch_first=True)
    music = pad_sequence([sample["music"] for sample in samples], batch_first=True)
    labels = pad_sequence([sample["labels"] for sample in samples], batch_first=True)
    frames = torch.arange(labels.shape[1])[None, :]
    padding_mask = frames >= lengths[:, None]
    return {
        "motion": motion,
        "music": music,
        "labels": labels,
        "lengths": lengths,
        "padding_mask": padding_mask,
        "names": [sample["name"] for sample in samples],
    }
