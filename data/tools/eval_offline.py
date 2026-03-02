"""Run offline evaluation from data/ side."""

from __future__ import annotations

import argparse
from pathlib import Path

from vtb.data.scan import filter_by_task_group, load_manifest
from vtb.eval.offline.eyeballing import run_offline_eyeballing
from vtb.eval.offline.maze import run_offline_maze
from vtb.eval.pipeline import write_eval_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="VideoThinkBench offline evaluation")
    parser.add_argument("--manifest", type=str, required=True)
    parser.add_argument("--task_group", type=str, required=True, choices=["maze", "eyeballing"])
    parser.add_argument("--pred_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser().resolve()
    pred_root = Path(args.pred_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    samples = filter_by_task_group(load_manifest(manifest_path), [args.task_group])
    if args.task_group == "maze":
        records = run_offline_maze(samples, pred_root)
    else:
        records = run_offline_eyeballing(samples, pred_root)

    summary = write_eval_outputs(output_dir, records)
    print("=" * 60)
    print("Offline evaluation finished")
    print(f"Results: {summary['results_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
