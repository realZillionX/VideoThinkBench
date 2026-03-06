from __future__ import annotations

import argparse
from pathlib import Path

from data.exporters.diffsynth_image import export_diffsynth_image
from data.exporters.diffsynth_video import export_diffsynth_video
from data.exporters.ms_swift import export_ms_swift
from data.scan import load_manifest


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("export", help="Export canonical manifest to training datasets")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument(
        "--target",
        type=str,
        required=True,
        choices=["ms-swift", "diffsynth-image", "diffsynth-video"],
    )
    parser.add_argument(
        "--task-groups",
        nargs="+",
        default=["eyeballing", "maze", "visual_puzzle"],
        choices=["eyeballing", "maze", "visual_puzzle"],
    )

    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for ms-swift target")
    parser.add_argument("--mode", type=str, default="sft,grpo", help="Modes for ms-swift target")
    parser.add_argument("--output", type=str, default=None, help="Output file path for diffsynth targets")

    parser.set_defaults(func=_cmd_export)


def _cmd_export(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    samples = load_manifest(manifest_path)

    if args.target == "ms-swift":
        if not args.output_dir:
            raise ValueError("--output-dir is required when --target ms-swift")
        output_dir = Path(args.output_dir).expanduser().resolve()
        modes = [part.strip() for part in args.mode.split(",") if part.strip()]
        outputs = export_ms_swift(
            samples,
            output_dir=output_dir,
            modes=modes,
            task_groups=args.task_groups,
        )
        print("=" * 60)
        print("VideoThinkBench data export finished")
        for out_path in outputs:
            print(f"Output: {out_path}")
        print("=" * 60)
        return

    if not args.output:
        raise ValueError("--output is required for diffsynth targets")
    output_path = Path(args.output).expanduser().resolve()

    if args.target == "diffsynth-image":
        out = export_diffsynth_image(samples, output_path=output_path, task_groups=args.task_groups)
    elif args.target == "diffsynth-video":
        out = export_diffsynth_video(samples, output_path=output_path, task_groups=args.task_groups)
    else:
        raise ValueError(f"Unsupported target: {args.target}")

    print("=" * 60)
    print("VideoThinkBench data export finished")
    print(f"Output: {out}")
    print("=" * 60)
