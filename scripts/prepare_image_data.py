"""Export DiffSynth image metadata from VideoThinkBench metadata."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.exporters.diffsynth_image import export_diffsynth_image
from data.scan import build_samples_from_data_root, load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VideoThinkBench datasets to DiffSynth image metadata")
    parser.add_argument("--dataset_root", type=str, default=None, help="Dataset root directory containing data.json files")
    parser.add_argument("--manifest", type=str, default=None, help="Canonical manifest path")
    parser.add_argument("--output_path", type=str, default="./dataset/metadata.json")
    parser.add_argument(
        "--task_groups",
        type=str,
        nargs="+",
        default=["eyeballing", "maze", "visual_puzzle"],
        choices=["eyeballing", "maze", "visual_puzzle"],
    )

    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        samples = load_manifest(manifest_path)
    else:
        if not args.dataset_root:
            raise ValueError("Either --manifest or --dataset_root must be provided")
        dataset_root = Path(args.dataset_root).expanduser().resolve()
        samples = build_samples_from_data_root(dataset_root, task_groups=args.task_groups)
        manifest_path = Path(args.output_path).expanduser().resolve().parent / "canonical_manifest.jsonl"
        write_manifest(manifest_path, samples)

    output_path = Path(args.output_path).expanduser().resolve()
    export_diffsynth_image(samples, output_path=output_path, task_groups=args.task_groups)

    print("=" * 60)
    print("VideoThinkBench -> DiffSynth image export finished")
    print(f"Manifest: {manifest_path}")
    print(f"Output: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
