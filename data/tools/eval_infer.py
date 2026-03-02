"""Run inference evaluation from data/ side."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtb.eval.commands import _run_infer
from vtb.eval.pipeline import write_eval_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoThinkBench inference evaluation")
    parser.add_argument("--modality", type=str, required=True, choices=["vlm", "image", "video"])
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mode", type=str, default="precheck", choices=["precheck", "validate"])
    parser.add_argument("--lora", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--diffsynth_path", type=str, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    args = parser.parse_args()

    translated = argparse.Namespace(
        modality=args.modality,
        dataset=args.dataset,
        model_path=args.model_path,
        output_dir=args.output_dir,
        mode=args.mode,
        lora=args.lora,
        num_samples=args.num_samples,
        device=args.device,
        diffsynth_path=args.diffsynth_path,
        dataset_root=args.dataset_root,
    )

    records = _run_infer(translated)
    summary = write_eval_outputs(Path(args.output_dir).expanduser().resolve(), records)
    print("=" * 60)
    print("Inference evaluation finished")
    print(f"Results: {summary['results_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
