"""Motion beat extraction from local minima of joint velocity."""

import numpy as np
from scipy.signal import argrelextrema


def extract_dance_beat_features(keypoints3d, fps=30):
    joints = np.asarray(keypoints3d)
    if joints.ndim != 3:
        raise ValueError("keypoints must have shape [frames, joints, 3]")
    velocity = np.zeros_like(joints, dtype=np.float32)
    velocity[1:] = joints[1:] - joints[:-1]
    envelope = np.linalg.norm(velocity, axis=2).sum(axis=1)
    order = max(1, int(round(fps / 6.0)))
    beat_indices = argrelextrema(envelope, np.less, axis=0, order=order)[0]
    beats = np.zeros_like(envelope, dtype=bool)
    beats[beat_indices] = True
    return beats
