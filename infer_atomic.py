"""Two-stage atomic planner/completion inference for AIST++ audio."""

import argparse
import json
import os
import pickle
import random
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/edge-numba-cache")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/edge-matplotlib-cache")

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.audio_extraction.baseline_features import FPS, SR, extract_audio
from dataset.atomic import labels_to_segments, refine_plan
from dataset.quaternion import ax_from_6v
from train_atomic import completion_model, planner_model, resolve_device
from vis import SMPLSkeleton


DEFAULT_PLANNER_CHECKPOINT = "runs/atomic_planner/planner.pt"
DEFAULT_COMPLETION_CHECKPOINT = "runs/atomic_completion/completion.pt"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_checkpoint(path, expected_stage, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("missing {} checkpoint: {}".format(expected_stage, path))
    try:
        checkpoint = torch.load(str(path), map_location="cpu", mmap=True)
    except TypeError:
        checkpoint = torch.load(str(path), map_location="cpu")
    if checkpoint.get("stage") != expected_stage:
        raise ValueError(
            "{} is a {} checkpoint, expected {}".format(
                path, checkpoint.get("stage"), expected_stage
            )
        )
    arguments = SimpleNamespace(**checkpoint["args"])
    model = planner_model(arguments) if expected_stage == "planner" else completion_model(arguments)
    model.load_state_dict(checkpoint["model"])
    model.to(device).eval()
    return model, arguments


class IndexedAtomicMotionLibrary:
    """Duration-aware atomic retrieval backed by memory-mapped training arrays."""

    def __init__(self, data_root):
        root = Path(data_root) / "train"
        self.motion = np.load(str(root / "motion.npy"), mmap_mode="r")
        self.labels = np.load(str(root / "labels.npy"), mmap_mode="r")
        if self.motion.shape[:2] != self.labels.shape:
            raise ValueError("training motion and label arrays are not aligned")
        self.index = {}
        for sample_index, labels in enumerate(self.labels):
            plan = torch.from_numpy(np.array(labels, dtype=np.int64, copy=True))
            for segment in labels_to_segments(plan):
                if segment.label:
                    self.index.setdefault(segment.label, []).append(
                        (sample_index, segment.start, segment.end)
                    )
        self._retrieval_cache = {}

    @property
    def labels_available(self):
        return set(self.index)

    def retrieve(self, label, target_length):
        key = (int(label), int(target_length))
        if key in self._retrieval_cache:
            return self._retrieval_cache[key].clone()
        candidates = self.index.get(int(label), ())
        if not candidates:
            raise KeyError("no training prototype for atomic label {}".format(label))
        sample, start, end = min(
            candidates, key=lambda item: abs((item[2] - item[1]) - target_length)
        )
        values = torch.from_numpy(np.array(self.motion[sample, start:end], copy=True))
        if len(values) != target_length:
            values = F.interpolate(
                values.T.unsqueeze(0),
                size=target_length,
                mode="linear",
                align_corners=True,
            ).squeeze(0).T
        self._retrieval_cache[key] = values
        return values.clone()

    def build_draft(self, labels, feature_dim):
        draft = torch.zeros(len(labels), feature_dim, dtype=torch.float32)
        mask = torch.zeros(len(labels), 1, dtype=torch.float32)
        for segment in labels_to_segments(labels):
            if segment.label == 0:
                continue
            draft[segment.start : segment.end] = self.retrieve(
                segment.label, segment.length
            )
            mask[segment.start : segment.end] = 1.0
        return draft, mask


class GroundTruthPlanStore:
    """Read frame-aligned oracle atomic labels from the preprocessed dataset."""

    def __init__(self, data_root):
        self.plans = {}
        for split in ("train", "test"):
            root = Path(data_root) / split
            with open(str(root / "names.json")) as handle:
                names = json.load(handle)
            labels = np.load(str(root / "labels.npy"), mmap_mode="r")
            if len(names) != len(labels):
                raise ValueError("unaligned names and labels under {}".format(root))
            for index, name in enumerate(names):
                if name in self.plans:
                    raise ValueError("duplicate atomic sample name: {}".format(name))
                self.plans[name] = (labels, index)

    def has_sequence(self, sequence_name):
        return sequence_name + "_slice0" in self.plans

    def get(self, sequence_name, length):
        key = sequence_name + "_slice0"
        if key not in self.plans:
            raise KeyError("no ground-truth atomic labels for {}".format(sequence_name))
        labels, index = self.plans[key]
        plan = torch.from_numpy(np.array(labels[index], dtype=np.int64, copy=True))
        if length > len(plan):
            raise ValueError(
                "ground-truth label slice for {} has only {} frames".format(
                    sequence_name, len(plan)
                )
            )
        return plan[:length]


def _window_starts(length, window_size, stride):
    if length <= window_size:
        return [0]
    starts = list(range(0, length - window_size + 1, stride))
    final = length - window_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _pad_frames(values, length):
    if len(values) >= length:
        return values[:length]
    padding = values.new_zeros((length - len(values),) + values.shape[1:])
    return torch.cat((values, padding), dim=0)


@torch.no_grad()
def infer_plan(
    planner,
    music,
    window_size,
    device,
    deterministic=False,
    temperature=1.0,
    vote_window=5,
    min_segment_length=6,
    available_labels=None,
):
    chunks = []
    lengths = []
    for start in range(0, len(music), window_size):
        chunk = music[start : start + window_size]
        lengths.append(len(chunk))
        chunks.append(_pad_frames(chunk, window_size))
    music_batch = torch.stack(chunks).to(device)
    padding_mask = torch.arange(window_size, device=device)[None] >= torch.tensor(
        lengths, device=device
    )[:, None]
    sampled = planner.sample(
        music_batch,
        padding_mask=padding_mask,
        temperature=temperature,
        deterministic=deterministic,
    ).cpu()
    labels = torch.cat([row[:length] for row, length in zip(sampled, lengths)])
    if available_labels is not None:
        unavailable = torch.ones_like(labels, dtype=torch.bool)
        for label in available_labels:
            unavailable &= labels != label
        unavailable &= labels != 0
        labels[unavailable] = 0
    return refine_plan(labels, vote_window, min_segment_length)


def _blend_weights(window_size, is_first, is_last, overlap):
    weights = torch.ones(window_size, 1)
    if overlap > 0 and not is_first:
        weights[:overlap] = torch.linspace(0.0, 1.0, overlap + 2)[1:-1, None]
    if overlap > 0 and not is_last:
        weights[-overlap:] = torch.linspace(1.0, 0.0, overlap + 2)[1:-1, None]
    return weights


@torch.no_grad()
def infer_completion(
    completion,
    music,
    draft,
    noise_mask,
    window_size,
    stride,
    device,
    guidance_weight=None,
    batch_size=1,
):
    if not 1 <= stride <= window_size:
        raise ValueError("completion stride must be in [1, window_size]")
    starts = _window_starts(len(music), window_size, stride)
    output = torch.zeros(len(music), draft.shape[1])
    weight_sum = torch.zeros(len(music), 1)
    overlap = window_size - stride
    if batch_size < 1:
        raise ValueError("inference batch size must be positive")
    batches = range(0, len(starts), batch_size)
    for batch_start in tqdm(
        batches,
        total=(len(starts) + batch_size - 1) // batch_size,
        desc="Completion",
        unit="batch",
    ):
        batch_starts = starts[batch_start : batch_start + batch_size]
        music_windows = []
        draft_windows = []
        mask_windows = []
        lengths = []
        for start in batch_starts:
            end = min(start + window_size, len(music))
            lengths.append(end - start)
            music_windows.append(_pad_frames(music[start:end], window_size))
            draft_windows.append(_pad_frames(draft[start:end], window_size))
            mask_windows.append(_pad_frames(noise_mask[start:end], window_size))
        generated_batch = completion.sample(
            torch.stack(music_windows).to(device),
            torch.stack(draft_windows).to(device),
            torch.stack(mask_windows).to(device),
            guidance_weight=guidance_weight,
        ).cpu()
        for offset, (start, valid_length, generated) in enumerate(
            zip(batch_starts, lengths, generated_batch)
        ):
            end = start + valid_length
            index = batch_start + offset
            weights = _blend_weights(
                valid_length,
                is_first=index == 0,
                is_last=index == len(starts) - 1,
                overlap=min(overlap, valid_length),
            )
            output[start:end] += generated[:valid_length] * weights
            weight_sum[start:end] += weights
    return output / weight_sum.clamp_min(1e-6)


def unnormalize_motion(motion, normalizer_path):
    normalizer = torch.load(str(normalizer_path), map_location="cpu")
    data_min = normalizer["data_min"].float()
    data_max = normalizer["data_max"].float()
    data_range = data_max - data_min
    safe_range = torch.where(
        data_range < 10 * torch.finfo(data_range.dtype).eps,
        torch.ones_like(data_range),
        data_range,
    )
    return (motion.clamp(-1.0, 1.0) + 1.0) * safe_range / 2.0 + data_min


def decode_motion(motion, normalizer_path):
    motion = unnormalize_motion(motion, normalizer_path)
    if motion.shape[1] != 151:
        raise ValueError("expected 151-D normalized motion, got {}".format(motion.shape[1]))
    contacts, values = torch.split(motion, (4, 147), dim=-1)
    root_positions = values[:, :3]
    rotations = ax_from_6v(values[:, 3:].reshape(-1, 24, 6))
    full_pose = SMPLSkeleton().forward(
        rotations.unsqueeze(0), root_positions.unsqueeze(0)
    )[0]
    return {
        "smpl_poses": rotations.reshape(-1, 72).numpy(),
        "smpl_trans": root_positions.numpy(),
        "full_pose": full_pose.numpy(),
        "contacts": contacts.numpy(),
    }


def _audio_map(audio_dir):
    mapping = {}
    for path in sorted(Path(audio_dir).rglob("*.wav")):
        if path.stem in mapping:
            raise ValueError("duplicate audio basename: {}".format(path.stem))
        mapping[path.stem] = path
    if not mapping:
        raise FileNotFoundError("no WAV files under {}".format(audio_dir))
    return mapping


def _aist_basename(name):
    fields = name.split("_")
    for index, field in enumerate(fields):
        if field.startswith("g") and len(field) == 3:
            return "_".join(fields[index:])
    return name


def _match_audio(name, audio):
    for candidate in (name, _aist_basename(name)):
        if candidate in audio:
            return audio[candidate]
    raise FileNotFoundError("no matching WAV for {}".format(name))


def _target_frames(path, raw_fps=60, output_fps=30):
    with open(str(path), "rb") as handle:
        data = pickle.load(handle)
    if "full_pose" in data:
        return len(data["full_pose"])
    for key in ("smpl_poses", "q"):
        if key in data:
            return int(round(len(data[key]) * output_fps / float(raw_fps)))
    raise KeyError("cannot determine motion length from {}".format(path))


def _load_music(path, frames=None):
    audio, _ = librosa.load(str(path), sr=SR)
    features = extract_audio(audio, path.stem, max_frames=None).astype(np.float32)
    if frames is not None:
        features = features[:frames]
        if len(features) < frames:
            features = np.pad(features, ((0, frames - len(features)), (0, 0)))
    return torch.from_numpy(features)


def _generated_is_current(
    path,
    planner_checkpoint,
    completion_checkpoint,
    expected_frames,
    plan_source,
    deterministic_planner,
    planner_temperature,
):
    try:
        with open(str(path), "rb") as handle:
            data = pickle.load(handle)
    except (OSError, EOFError, pickle.UnpicklingError):
        return False
    return (
        data.get("planner_checkpoint") == str(planner_checkpoint)
        and data.get("completion_checkpoint") == str(completion_checkpoint)
        and data.get("plan_source", "planner") == plan_source
        and data.get("deterministic_planner") == deterministic_planner
        and data.get("planner_temperature") == planner_temperature
        and "full_pose" in data
        and (expected_frames is None or len(data["full_pose"]) == expected_frames)
    )


def _filter_and_refine_labels(
    labels, available_labels, vote_window=5, min_segment_length=6
):
    labels = labels.clone()
    unavailable = torch.ones_like(labels, dtype=torch.bool)
    for label in available_labels:
        unavailable &= labels != label
    unavailable &= labels != 0
    labels[unavailable] = 0
    return refine_plan(labels, vote_window, min_segment_length)


def _write_generated_result(
    output_path,
    normalized,
    labels,
    audio_path,
    normalizer_path,
    planner_checkpoint,
    completion_checkpoint,
    plan_source="planner",
    deterministic_planner=False,
    planner_temperature=1.0,
):
    result = decode_motion(normalized, normalizer_path)
    result["atomic_labels"] = labels.numpy()
    result["audio_path"] = str(audio_path)
    result["planner_checkpoint"] = str(planner_checkpoint)
    result["completion_checkpoint"] = str(completion_checkpoint)
    result["plan_source"] = plan_source
    result["deterministic_planner"] = deterministic_planner
    result["planner_temperature"] = planner_temperature
    with open(str(output_path), "wb") as handle:
        pickle.dump(result, handle)


def infer_directory(
    audio_dir,
    output_dir,
    planner_checkpoint=DEFAULT_PLANNER_CHECKPOINT,
    completion_checkpoint=DEFAULT_COMPLETION_CHECKPOINT,
    data_root="data/atomic_aistpp",
    target_motion_dir=None,
    device="auto",
    seed=42,
    max_samples=None,
    max_frames=None,
    overwrite=False,
    deterministic_planner=False,
    temperature=1.0,
    completion_stride=75,
    guidance_weight=None,
    draft_noise_ratio=None,
    inference_batch_size=4,
    sequence_names=None,
    ground_truth_labels=False,
):
    seed_everything(seed)
    if inference_batch_size < 1:
        raise ValueError("inference batch size must be positive")
    device = resolve_device(device)
    completion, completion_args = _load_checkpoint(
        completion_checkpoint, "completion", device
    )
    if ground_truth_labels:
        planner = None
        planner_args = completion_args
        ground_truth_plans = GroundTruthPlanStore(data_root)
    else:
        planner, planner_args = _load_checkpoint(planner_checkpoint, "planner", device)
        ground_truth_plans = None
        if planner_args.music_dim != completion_args.music_dim:
            raise ValueError("planner and completion music dimensions differ")
        if planner_args.seq_len != completion_args.seq_len:
            raise ValueError("planner and completion window lengths differ")
    library = IndexedAtomicMotionLibrary(data_root)
    normalizer_path = Path(data_root) / "normalizer.pt"
    if not normalizer_path.is_file():
        raise FileNotFoundError("missing motion normalizer: {}".format(normalizer_path))

    audio = _audio_map(audio_dir)
    if target_motion_dir:
        target_map = {
            path.stem: path for path in sorted(Path(target_motion_dir).glob("*.pkl"))
        }
        if sequence_names is None:
            targets = list(target_map.values())
        else:
            missing = [name for name in sequence_names if name not in target_map]
            if missing:
                raise FileNotFoundError(
                    "sequence list motions are missing: {}".format(missing[:5])
                )
            targets = [target_map[name] for name in sequence_names]
        items = [
            (path.stem, _match_audio(path.stem, audio), _target_frames(path))
            for path in targets
        ]
    else:
        names = sorted(audio) if sequence_names is None else sequence_names
        missing = [name for name in names if name not in audio]
        if missing:
            raise FileNotFoundError(
                "sequence list audio files are missing: {}".format(missing[:5])
            )
        items = [(name, audio[name], None) for name in names]
    if max_samples is not None:
        items = items[:max_samples]
    if ground_truth_plans is not None:
        original_count = len(items)
        items = [item for item in items if ground_truth_plans.has_sequence(item[0])]
        if len(items) != original_count:
            print(
                "Skipping {} sequences without ground-truth atomic labels".format(
                    original_count - len(items)
                )
            )
    if not items:
        raise FileNotFoundError("no inference inputs found")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_paths = []
    ratio = (
        completion_args.draft_noise_ratio
        if draft_noise_ratio is None
        else draft_noise_ratio
    )
    plan_source = "ground_truth" if ground_truth_labels else "planner"
    pending = []
    for index, (name, audio_path, target_frames) in enumerate(items):
        output_path = output_dir / (name + ".pkl")
        generated_paths.append(output_path)
        frames = target_frames
        if max_frames is not None:
            frames = max_frames if frames is None else min(frames, max_frames)
        if (
            output_path.is_file()
            and not overwrite
            and _generated_is_current(
                output_path,
                planner_checkpoint,
                completion_checkpoint,
                frames,
                plan_source,
                deterministic_planner,
                temperature,
            )
        ):
            print("Using generated motion: {}".format(output_path))
            continue
        pending.append((index, name, audio_path, output_path, frames))

    short_batch_mode = pending and all(
        frames is not None and frames <= planner_args.seq_len
        for _, _, _, _, frames in pending
    )
    if short_batch_mode:
        batches = range(0, len(pending), inference_batch_size)
        for batch_start in tqdm(
            batches,
            total=(len(pending) + inference_batch_size - 1) // inference_batch_size,
            desc=(
                "Completion inference"
                if ground_truth_plans is not None
                else "Two-stage inference"
            ),
            unit="batch",
        ):
            batch = pending[batch_start : batch_start + inference_batch_size]
            music_rows = [_load_music(row[2], row[4]) for row in batch]
            lengths = [len(music) for music in music_rows]
            if min(lengths) < 3:
                raise ValueError("audio is too short for inference")
            music_batch = torch.stack(
                [_pad_frames(music, planner_args.seq_len) for music in music_rows]
            ).to(device)
            padding_mask = torch.arange(planner_args.seq_len, device=device)[None] >= torch.tensor(
                lengths, device=device
            )[:, None]
            seed_everything(seed + batch[0][0])
            if ground_truth_plans is not None:
                plans = [
                    ground_truth_plans.get(row[1], length)
                    for row, length in zip(batch, lengths)
                ]
            else:
                sampled = planner.sample(
                    music_batch,
                    padding_mask=padding_mask,
                    temperature=temperature,
                    deterministic=deterministic_planner,
                ).cpu()
                plans = [
                    _filter_and_refine_labels(
                        labels[:length], library.labels_available
                    )
                    for labels, length in zip(sampled, lengths)
                ]
            conditions = [
                library.build_draft(labels, completion_args.motion_dim)
                for labels in plans
            ]
            draft_batch = torch.stack(
                [_pad_frames(draft, completion_args.seq_len) for draft, _ in conditions]
            ).to(device)
            mask_batch = torch.stack(
                [_pad_frames(mask * ratio, completion_args.seq_len) for _, mask in conditions]
            ).to(device)
            normalized_batch = completion.sample(
                music_batch,
                draft_batch,
                mask_batch,
                guidance_weight=guidance_weight,
            ).cpu()
            for row, length, labels, normalized in zip(
                batch, lengths, plans, normalized_batch
            ):
                _, _, audio_path, output_path, _ = row
                _write_generated_result(
                    output_path,
                    normalized[:length],
                    labels,
                    audio_path,
                    normalizer_path,
                    planner_checkpoint,
                    completion_checkpoint,
                    plan_source,
                    deterministic_planner,
                    temperature,
                )
    else:
        if ground_truth_plans is not None:
            raise ValueError(
                "ground-truth-label inference currently supports at most one 150-frame slice"
            )
        for index, name, audio_path, output_path, frames in pending:
            music = _load_music(audio_path, frames)
            if len(music) < 3:
                raise ValueError("audio is too short for inference: {}".format(audio_path))
            seed_everything(seed + index)
            labels = infer_plan(
                planner,
                music,
                planner_args.seq_len,
                device,
                deterministic=deterministic_planner,
                temperature=temperature,
                available_labels=library.labels_available,
            )
            draft, noise_mask = library.build_draft(labels, completion_args.motion_dim)
            normalized = infer_completion(
                completion,
                music,
                draft,
                noise_mask * ratio,
                completion_args.seq_len,
                completion_stride,
                device,
                guidance_weight,
                inference_batch_size,
            )
            _write_generated_result(
                output_path,
                normalized,
                labels,
                audio_path,
                normalizer_path,
                planner_checkpoint,
                completion_checkpoint,
                plan_source,
                deterministic_planner,
                temperature,
            )

    manifest = {
        "samples": len(generated_paths),
        "names": [path.stem for path in generated_paths],
        "output_dir": str(output_dir),
        "planner_checkpoint": str(planner_checkpoint),
        "completion_checkpoint": str(completion_checkpoint),
        "device": str(device),
        "plan_source": plan_source,
        "deterministic_planner": deterministic_planner,
        "planner_temperature": temperature,
    }
    with open(str(output_dir / "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args():
    parser = argparse.ArgumentParser(description="Run two-stage atomic dance inference")
    parser.add_argument("--audio-dir", required=True)
    parser.add_argument("--output-dir", default="eval/generated_motions")
    parser.add_argument("--target-motion-dir")
    parser.add_argument("--planner-checkpoint", default=DEFAULT_PLANNER_CHECKPOINT)
    parser.add_argument("--completion-checkpoint", default=DEFAULT_COMPLETION_CHECKPOINT)
    parser.add_argument("--data-root", default="data/atomic_aistpp")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--overwrite", action="store_true")
    sampler = parser.add_mutually_exclusive_group()
    sampler.add_argument(
        "--deterministic-planner",
        action="store_true",
        help="use argmax reverse steps for debugging instead of D3PM sampling",
    )
    sampler.add_argument(
        "--stochastic-planner",
        dest="deterministic_planner",
        action="store_false",
        help="sample every categorical reverse step (default)",
    )
    parser.set_defaults(deterministic_planner=False)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--completion-stride", type=int, default=75)
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--draft-noise-ratio", type=float)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--sequence-list")
    parser.add_argument(
        "--plan-source",
        choices=("planner", "ground-truth"),
        default="planner",
    )
    parser.add_argument("--ground-truth-labels", action="store_true")
    return parser.parse_args()


def main(options):
    sequence_names = None
    if options.sequence_list:
        sequence_names = [
            line.strip()
            for line in Path(options.sequence_list).read_text().splitlines()
            if line.strip()
        ]
    manifest = infer_directory(
        audio_dir=options.audio_dir,
        output_dir=options.output_dir,
        planner_checkpoint=options.planner_checkpoint,
        completion_checkpoint=options.completion_checkpoint,
        data_root=options.data_root,
        target_motion_dir=options.target_motion_dir,
        device=options.device,
        seed=options.seed,
        max_samples=options.max_samples,
        max_frames=options.max_frames,
        overwrite=options.overwrite,
        deterministic_planner=options.deterministic_planner,
        temperature=options.temperature,
        completion_stride=options.completion_stride,
        guidance_weight=options.guidance_weight,
        draft_noise_ratio=options.draft_noise_ratio,
        inference_batch_size=options.inference_batch_size,
        sequence_names=sequence_names,
        ground_truth_labels=(
            options.ground_truth_labels or options.plan_source == "ground-truth"
        ),
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main(parse_args())
