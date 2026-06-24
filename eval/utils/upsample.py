"""Utilities for interpolating generated SMPL data from 30 to 60 FPS."""

import argparse
import pickle
from pathlib import Path

import torch
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle


def slerp(first, second, amount):
    first = first / torch.linalg.norm(first, dim=-1, keepdim=True)
    second = second / torch.linalg.norm(second, dim=-1, keepdim=True)
    dot = torch.sum(first * second, dim=-1, keepdim=True)
    second = torch.where(dot < 0.0, -second, second)
    dot = torch.abs(dot).clamp(max=1.0)
    use_lerp = dot > 0.9995
    theta = torch.acos(dot)
    sine = torch.sin(theta).clamp(min=1e-6)
    interpolated = (
        (torch.cos(theta * amount) - dot * torch.sin(theta * amount) / sine)
        * first
        + torch.sin(theta * amount) / sine * second
    )
    linear = first + amount * (second - first)
    result = torch.where(use_lerp, linear, interpolated)
    return result / torch.linalg.norm(result, dim=-1, keepdim=True).clamp(min=1e-6)


def _interleave(original, intermediate):
    result = torch.empty(
        (original.shape[0] * 2 - 1,) + original.shape[1:], dtype=original.dtype
    )
    result[::2] = original
    result[1::2] = intermediate
    return result


def upsample_smpl_data(input_path, output_path):
    with open(str(input_path), "rb") as handle:
        data = pickle.load(handle)
    required = ("smpl_poses", "smpl_trans", "full_pose")
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError("missing fields: {}".format(", ".join(missing)))

    poses = torch.as_tensor(data["smpl_poses"], dtype=torch.float32)
    translations = torch.as_tensor(data["smpl_trans"], dtype=torch.float32)
    positions = torch.as_tensor(data["full_pose"], dtype=torch.float32)
    if len(poses) < 2 or len(translations) != len(poses) or len(positions) != len(poses):
        raise ValueError("motion fields must have the same length and at least two frames")

    rotations = poses.reshape(len(poses), -1, 3)
    quaternions = axis_angle_to_quaternion(rotations)
    middle_poses = quaternion_to_axis_angle(slerp(quaternions[:-1], quaternions[1:], 0.5))
    output_data = dict(data)
    output_data["smpl_poses"] = _interleave(
        poses, middle_poses.reshape(len(poses) - 1, -1)
    ).numpy()
    output_data["smpl_trans"] = _interleave(
        translations, (translations[:-1] + translations[1:]) / 2
    ).numpy()
    output_data["full_pose"] = _interleave(
        positions, (positions[:-1] + positions[1:]) / 2
    ).numpy()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "wb") as handle:
        pickle.dump(output_data, handle)


def batch_upsample(input_dir, output_dir):
    paths = sorted(Path(input_dir).glob("*.pkl"))
    if not paths:
        raise FileNotFoundError("no motion PKLs under {}".format(input_dir))
    for path in paths:
        upsample_smpl_data(path, Path(output_dir) / path.name)
    return len(paths)


def main():
    parser = argparse.ArgumentParser(description="Upsample EDGE SMPL PKLs to 60 FPS")
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")
    options = parser.parse_args()
    print("processed={}".format(batch_upsample(options.input_dir, options.output_dir)))


if __name__ == "__main__":
    main()
