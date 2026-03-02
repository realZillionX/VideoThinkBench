"""Export ms-swift datasets from VideoThinkBench metadata.

This script keeps the historical command path while using the unified
manifest/export pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vtb.data.exporters.ms_swift import export_ms_swift
from vtb.data.scan import build_samples_from_data_root, load_manifest, write_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert VideoThinkBench datasets to ms-swift JSONL format")
    parser.add_argument("--data_root", type=str, default=None, help="Dataset root directory containing data.json files")
    parser.add_argument("--manifest", type=str, default=None, help="Canonical manifest path")
    parser.add_argument("--output_dir", type=str, default=".", help="Output directory for JSONL files")
    parser.add_argument("--task_groups", type=str, nargs="+", default=["eyeballing", "maze"], choices=["eyeballing", "maze"])

    args = parser.parse_args()

    if args.manifest:
        manifest_path = Path(args.manifest).expanduser().resolve()
        samples = load_manifest(manifest_path)
    else:
        if not args.data_root:
            raise ValueError("Either --manifest or --data_root must be provided")
        data_root = Path(args.data_root).expanduser().resolve()
        samples = build_samples_from_data_root(data_root, task_groups=args.task_groups)
        manifest_path = Path(args.output_dir).expanduser().resolve() / "canonical_manifest.jsonl"
        write_manifest(manifest_path, samples)

    output_dir = Path(args.output_dir).expanduser().resolve()
    outputs = export_ms_swift(
        samples,
        output_dir=output_dir,
        modes=["sft", "grpo"],
        task_groups=args.task_groups,
    )

    print("=" * 60)
    print("VideoThinkBench -> ms-swift export finished")
    print(f"Manifest: {manifest_path}")
    for output in outputs:
        print(f"Output: {output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
