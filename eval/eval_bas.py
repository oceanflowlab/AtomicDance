"""Beat alignment score (BAS) for extracted music and dance beats."""

from pathlib import Path

import numpy as np



def alignment_score(music_beats, motion_beats, sigma=3.0):
    """Return the project-scaled Gaussian beat-alignment score."""
    music_indices = np.flatnonzero(np.asarray(music_beats).reshape(-1))
    motion_indices = np.flatnonzero(np.asarray(motion_beats).reshape(-1))
    if len(music_indices) == 0 or len(motion_indices) == 0:
        return 0.0
    distances = np.abs(motion_indices[:, None] - music_indices[None, :]).min(axis=1)
    raw_score = 0.5 * np.exp(
        -(distances.astype(np.float64) ** 2) / (2.0 * sigma ** 2)
    ).mean()
    return float(raw_score)


def _npy_files(directory):
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError("missing feature directory: {}".format(directory))
    return {path.stem: path for path in directory.glob("*.npy")}


def calculate_bas(root, sigma=3.0, return_per_sequence=False):
    root = Path(root)
    music = _npy_files(root / "music_features")
    dance = _npy_files(root / "dance_features")
    if set(music) != set(dance):
        missing_music = sorted(set(dance) - set(music))
        missing_dance = sorted(set(music) - set(dance))
        raise ValueError(
            "unmatched BAS features; missing music={}, missing dance={}".format(
                missing_music[:5], missing_dance[:5]
            )
        )
    if not music:
        raise ValueError("no BAS features found under {}".format(root))
    scores = {
        name: alignment_score(np.load(str(music[name])), np.load(str(dance[name])), sigma)
        for name in sorted(music)
    }
    mean = float(np.mean(list(scores.values())))
    return (mean, scores) if return_per_sequence else mean


def calculate_BAS(pkl_root):
    """Backward-compatible wrapper used by the starter script."""
    return calculate_bas(pkl_root)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("feature_root")
    parser.add_argument("--sigma", type=float, default=3.0)
    options = parser.parse_args()
    print(calculate_bas(options.feature_root, sigma=options.sigma))
