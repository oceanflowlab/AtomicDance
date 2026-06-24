"""Compatibility entry point for AIST++ evaluation feature extraction."""

import json

from eval.extract_aist_features import extract_directory, parse_args


def main(options):
    manifest = extract_directory(
        motion_dir=options.motion_dir,
        audio_dir=options.audio_dir,
        output_root=options.output,
        smpl_model=options.smpl_model,
        workers=options.workers,
        evaluation_fps=options.evaluation_fps,
        generated_fps=options.generated_fps,
        raw_fps=options.raw_fps,
        max_frames=options.max_frames,
    )
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    main(parse_args())
