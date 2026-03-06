from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from data.scan import filter_by_task_group, load_manifest
from evaluators.infer.image import run_image_infer
from evaluators.infer.video import run_video_infer
from evaluators.infer.vlm import run_vlm_infer
from evaluators.offline.eyeballing import run_offline_eyeballing
from evaluators.offline.maze import run_offline_maze
from evaluators.offline.visual_puzzle import run_offline_visual_puzzle
from evaluators.pipeline import write_eval_outputs
from utils.schemas import EvalRecord


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = parser.add_subparsers(dest="eval_command", required=True)

    offline = eval_subparsers.add_parser("offline", help="Run offline evaluators")
    offline.add_argument("--manifest", type=str, required=True)
    offline.add_argument("--task-group", type=str, required=True, choices=["maze", "eyeballing", "visual_puzzle"])
    offline.add_argument("--pred-root", type=str, required=True)
    offline.add_argument("--output-dir", type=str, required=True)
    offline.set_defaults(func=_cmd_eval_offline)

    infer = eval_subparsers.add_parser("infer", help="Run model inference evaluators")
    infer.add_argument("--modality", type=str, required=True, choices=["vlm", "image", "video"])
    infer.add_argument("--dataset", type=str, required=True)
    infer.add_argument("--model-path", type=str, required=True)
    infer.add_argument("--output-dir", type=str, required=True)
    infer.add_argument("--mode", type=str, default="precheck", choices=["precheck", "validate"])
    infer.add_argument("--lora", type=str, default=None)
    infer.add_argument("--num-samples", type=int, default=None)
    infer.add_argument("--device", type=str, default="cuda")
    infer.add_argument("--diffsynth-path", type=str, default=None)
    infer.add_argument("--dataset-root", type=str, default=None)
    infer.set_defaults(func=_cmd_eval_infer)

    run = eval_subparsers.add_parser("run", help="Run inference and optional offline evaluation")
    run.add_argument("--modality", type=str, required=True, choices=["vlm", "image", "video"])
    run.add_argument("--dataset", type=str, required=True)
    run.add_argument("--model-path", type=str, required=True)
    run.add_argument("--output-dir", type=str, required=True)
    run.add_argument("--mode", type=str, default="validate", choices=["precheck", "validate"])
    run.add_argument("--lora", type=str, default=None)
    run.add_argument("--num-samples", type=int, default=None)
    run.add_argument("--device", type=str, default="cuda")
    run.add_argument("--diffsynth-path", type=str, default=None)
    run.add_argument("--dataset-root", type=str, default=None)
    run.add_argument("--with-offline", action="store_true")
    run.add_argument("--manifest", type=str, default=None)
    run.set_defaults(func=_cmd_eval_run)


def _run_offline(task_group: str, manifest_path: Path, pred_root: Path) -> List[EvalRecord]:
    samples = load_manifest(manifest_path)
    target_samples = filter_by_task_group(samples, [task_group])
    if task_group == "maze":
        return run_offline_maze(target_samples, pred_root)
    if task_group == "visual_puzzle":
        return run_offline_visual_puzzle(target_samples, pred_root)
    return run_offline_eyeballing(target_samples, pred_root)


def _cmd_eval_offline(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser().resolve()
    pred_root = Path(args.pred_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    records = _run_offline(args.task_group, manifest_path, pred_root)
    summary = write_eval_outputs(output_dir, records)

    print("=" * 60)
    print("VideoThinkBench eval offline finished")
    print(f"Results: {summary['results_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("=" * 60)


def _run_infer(args: argparse.Namespace) -> List[EvalRecord]:
    dataset_path = Path(args.dataset).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    if args.modality == "vlm":
        return run_vlm_infer(
            dataset_path=dataset_path,
            model_path=args.model_path,
            output_dir=output_dir,
            device=args.device,
            mode=args.mode,
            num_samples=args.num_samples,
        )

    if args.modality == "image":
        return run_image_infer(
            dataset_path=dataset_path,
            model_path=args.model_path,
            output_dir=output_dir,
            mode=args.mode,
            lora_path=args.lora,
            num_samples=args.num_samples,
            diffsynth_path=args.diffsynth_path,
        )

    dataset_root = Path(args.dataset_root).expanduser().resolve() if args.dataset_root else None
    return run_video_infer(
        dataset_path=dataset_path,
        model_path=args.model_path,
        output_dir=output_dir,
        mode=args.mode,
        lora_path=args.lora,
        num_samples=args.num_samples,
        diffsynth_path=args.diffsynth_path,
        dataset_root=dataset_root,
        device=args.device,
    )


def _cmd_eval_infer(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = _run_infer(args)
    summary = write_eval_outputs(output_dir, records)

    print("=" * 60)
    print("VideoThinkBench eval infer finished")
    print(f"Results: {summary['results_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("=" * 60)


def _merge_offline_into_infer(infer_records: List[EvalRecord], offline_records: List[EvalRecord]) -> List[EvalRecord]:
    by_id: Dict[str, EvalRecord] = {record.sample_id: record for record in infer_records}
    for off in offline_records:
        target = by_id.get(off.sample_id)
        if target is None:
            continue
        target.offline_metrics = off.offline_metrics
        target.offline_pass = off.offline_pass
        if off.error:
            target.error = off.error
        if off.prediction_text:
            target.prediction_text = off.prediction_text
    return infer_records


def _cmd_eval_run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir).expanduser().resolve()
    records = _run_infer(args)

    if args.with_offline:
        if not args.manifest:
            raise ValueError("--manifest is required when --with-offline is enabled")
        manifest_path = Path(args.manifest).expanduser().resolve()
        pred_root = output_dir / "samples"

        samples = load_manifest(manifest_path)
        groups = sorted({sample.task_group for sample in samples})
        offline_records: List[EvalRecord] = []
        for group in groups:
            if group == "maze":
                offline_records.extend(run_offline_maze(filter_by_task_group(samples, [group]), pred_root))
            elif group == "eyeballing":
                offline_records.extend(run_offline_eyeballing(filter_by_task_group(samples, [group]), pred_root))
            elif group == "visual_puzzle":
                offline_records.extend(run_offline_visual_puzzle(filter_by_task_group(samples, [group]), pred_root))
        records = _merge_offline_into_infer(records, offline_records)

    summary = write_eval_outputs(output_dir, records)

    print("=" * 60)
    print("VideoThinkBench eval run finished")
    print(f"Results: {summary['results_path']}")
    print(f"Summary: {summary['summary_path']}")
    print("=" * 60)
