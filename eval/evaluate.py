"""Unified EDGE/AIST++ evaluation entry point."""

import argparse
import json
from pathlib import Path

from eval.extract_aist_features import DEFAULT_SMPL_MODEL, extract_directory
from eval.metrics import quantized_metrics
from infer_atomic import (
    DEFAULT_COMPLETION_CHECKPOINT,
    DEFAULT_PLANNER_CHECKPOINT,
    infer_directory,
)


def evaluate(
    prediction_features,
    ground_truth_features,
):
    return quantized_metrics(prediction_features, ground_truth_features)


def _motion_names(motion_dir, include_names=None):
    names = [path.stem for path in sorted(Path(motion_dir).glob("*.pkl"))]
    if include_names is not None:
        requested = set(include_names)
        names = [name for name in names if name in requested]
    return names


def _motion_signature(motion_dir, include_names=None):
    names = set(_motion_names(motion_dir, include_names))
    return [
        [path.stem, path.stat().st_size, path.stat().st_mtime_ns]
        for path in sorted(Path(motion_dir).glob("*.pkl"))
        if path.stem in names
    ]


def _cache_is_current(
    feature_root,
    motion_dir,
    audio_dir,
    smpl_model,
    evaluation_fps,
    generated_fps,
    raw_fps,
    include_names=None,
    max_frames=None,
):
    manifest_path = Path(feature_root) / "manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        with open(str(manifest_path)) as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return False
    expected = {
        "names": _motion_names(motion_dir, include_names),
        "evaluation_fps": evaluation_fps,
        "generated_fps": generated_fps,
        "raw_fps": raw_fps,
        "motion_dir": str(motion_dir),
        "audio_dir": str(audio_dir),
        "smpl_model": smpl_model,
        "max_frames": max_frames,
        "motion_signature": _motion_signature(motion_dir, include_names),
    }
    return all(manifest.get(key) == value for key, value in expected.items())


def resolve_features(
    label,
    feature_root,
    motion_dir,
    audio_dir,
    cache_dir,
    smpl_model=None,
    workers=1,
    evaluation_fps=30,
    generated_fps=30,
    raw_fps=60,
    force=False,
    include_names=None,
    max_frames=None,
):
    """Return an existing feature root or extract one from motions and audio."""
    if feature_root:
        return Path(feature_root)
    if not motion_dir:
        raise ValueError(
            "--{}-motions is required when --{}-features is omitted".format(
                label, label
            )
        )
    if not audio_dir:
        raise ValueError(
            "an audio directory is required to extract {} features".format(label)
        )
    feature_root = Path(cache_dir) / label
    if force or not _cache_is_current(
        feature_root,
        motion_dir,
        audio_dir,
        smpl_model,
        evaluation_fps,
        generated_fps,
        raw_fps,
        include_names,
        max_frames,
    ):
        extract_directory(
            motion_dir=motion_dir,
            audio_dir=audio_dir,
            output_root=feature_root,
            smpl_model=smpl_model,
            workers=workers,
            evaluation_fps=evaluation_fps,
            generated_fps=generated_fps,
            raw_fps=raw_fps,
            include_names=include_names,
            max_frames=max_frames,
        )
    else:
        print("Using cached {} features: {}".format(label, feature_root))
    return feature_root


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate generated AIST++ dances")
    parser.add_argument("--prediction-features")
    parser.add_argument("--ground-truth-features")
    parser.add_argument("--prediction-motions")
    parser.add_argument("--ground-truth-motions")
    parser.add_argument(
        "--audio-dir", help="shared audio directory for prediction and ground truth"
    )
    parser.add_argument("--prediction-audio-dir")
    parser.add_argument("--ground-truth-audio-dir")
    parser.add_argument("--smpl-model", default=DEFAULT_SMPL_MODEL)
    parser.add_argument("--cache-dir", default="eval/cache")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--evaluation-fps", type=int, default=30)
    parser.add_argument("--generated-fps", type=int, default=30)
    parser.add_argument("--raw-fps", type=int, default=60)
    parser.add_argument("--force-extract", action="store_true")
    parser.add_argument("--planner-checkpoint", default=DEFAULT_PLANNER_CHECKPOINT)
    parser.add_argument("--completion-checkpoint", default=DEFAULT_COMPLETION_CHECKPOINT)
    parser.add_argument("--atomic-data-root", default="data/atomic_aistpp")
    parser.add_argument("--inference-output", default="eval/generated_motions")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--max-inference-samples", type=int)
    parser.add_argument("--max-inference-frames", type=int)
    parser.add_argument("--overwrite-inference", action="store_true")
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
    parser.add_argument("--planner-temperature", type=float, default=1.0)
    parser.add_argument("--completion-stride", type=int, default=75)
    parser.add_argument("--guidance-weight", type=float)
    parser.add_argument("--draft-noise-ratio", type=float)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument("--sequence-list")
    parser.add_argument(
        "--plan-source",
        choices=("planner", "ground-truth"),
        default="planner",
        help="use predicted planner labels or oracle labels for completion",
    )
    parser.add_argument("--ground-truth-labels", action="store_true")
    parser.add_argument("--output", default="eval/results.json")
    return parser.parse_args()


def main(options):
    prediction_audio = options.prediction_audio_dir or options.audio_dir
    ground_truth_audio = options.ground_truth_audio_dir or options.audio_dir
    prediction_motions = options.prediction_motions
    inferred_names = None
    sequence_names = None
    if options.sequence_list:
        sequence_names = [
            line.strip()
            for line in Path(options.sequence_list).read_text().splitlines()
            if line.strip()
        ]
    if not prediction_motions and not options.prediction_features:
        if not options.ground_truth_motions:
            raise ValueError(
                "--ground-truth-motions is required for automatic inference"
            )
        if not prediction_audio:
            raise ValueError("an audio directory is required for automatic inference")
        inference_manifest = infer_directory(
            audio_dir=prediction_audio,
            output_dir=options.inference_output,
            planner_checkpoint=options.planner_checkpoint,
            completion_checkpoint=options.completion_checkpoint,
            data_root=options.atomic_data_root,
            target_motion_dir=options.ground_truth_motions,
            device=options.device,
            seed=options.inference_seed,
            max_samples=options.max_inference_samples,
            max_frames=options.max_inference_frames,
            overwrite=options.overwrite_inference,
            deterministic_planner=options.deterministic_planner,
            temperature=options.planner_temperature,
            completion_stride=options.completion_stride,
            guidance_weight=options.guidance_weight,
            draft_noise_ratio=options.draft_noise_ratio,
            inference_batch_size=options.inference_batch_size,
            sequence_names=sequence_names,
            ground_truth_labels=(
                options.ground_truth_labels
                or options.plan_source == "ground-truth"
            ),
        )
        prediction_motions = options.inference_output
        inferred_names = inference_manifest["names"]
    evaluation_names = inferred_names or sequence_names
    prediction_features = resolve_features(
        "prediction",
        options.prediction_features,
        prediction_motions,
        prediction_audio,
        options.cache_dir,
        options.smpl_model,
        options.workers,
        options.evaluation_fps,
        options.generated_fps,
        options.raw_fps,
        options.force_extract,
        evaluation_names,
        options.max_inference_frames if inferred_names is not None else None,
    )
    ground_truth_features = resolve_features(
        "ground_truth",
        options.ground_truth_features,
        options.ground_truth_motions,
        ground_truth_audio,
        options.cache_dir,
        options.smpl_model,
        options.workers,
        options.evaluation_fps,
        options.generated_fps,
        options.raw_fps,
        options.force_extract,
        evaluation_names,
        options.max_inference_frames if inferred_names is not None else None,
    )
    metrics = evaluate(prediction_features, ground_truth_features)
    output = Path(options.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output), "w") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print("results={}".format(output))
    return metrics


if __name__ == "__main__":
    main(parse_args())
