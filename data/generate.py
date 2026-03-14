from __future__ import annotations

import argparse
import importlib
import inspect
import json
import multiprocessing as mp
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from data.scan import build_canonical_sample, dedup_samples, write_manifest
from data.registry import TASK_SPECS, TaskSpec, resolve_requested_tasks
from core.io import read_json, write_json


def _build_visual_puzzle_ti2v_prompt(sample: Dict[str, Any], instruction: str, common_instruction: str) -> str:
    question = str(sample.get("question") or "").strip()
    parts = [
        "Use the provided visual puzzle image as the starting frame.",
        question,
        instruction.strip(),
        common_instruction.strip(),
    ]
    return " ".join(part for part in parts if part).strip()


def _build_visual_puzzle_ti2t_prompt(sample: Dict[str, Any]) -> str:
    question = str(sample.get("question") or "").strip()
    options = sample.get("options") or []
    option_text = ", ".join(str(option).strip() for option in options if str(option).strip())
    if option_text:
        return f"Use the provided visual puzzle image to solve the task. {question} Choose from: {option_text}. Answer with the final option text only."
    return f"Use the provided visual puzzle image to solve the task. {question} Answer with the final answer only."


def load_generator_class(spec: TaskSpec):
    module = importlib.import_module(spec.module)
    if spec.class_name and hasattr(module, spec.class_name):
        return getattr(module, spec.class_name)
    for attr_name in dir(module):
        attr = getattr(module, attr_name)
        if isinstance(attr, type) and attr_name.endswith("Generator"):
            return attr
    raise ImportError(f"Generator class not found in module: {spec.module}")


def parse_task_config(config_path: Optional[str], config_json: Optional[str]) -> Dict[str, Dict[str, Any]]:
    if config_path:
        payload = read_json(Path(config_path))
        if not isinstance(payload, dict):
            raise ValueError("task_config_path must contain JSON object")
        return payload
    if config_json:
        payload = json.loads(config_json)
        if not isinstance(payload, dict):
            raise ValueError("task_config must be a JSON object")
        return payload
    return {}


def split_counts(total: int, parts: int) -> List[int]:
    if parts <= 0:
        return []
    base = total // parts
    extra = total % parts
    return [base + (1 if idx < extra else 0) for idx in range(parts)]


def build_default_kwargs(spec: TaskSpec, args: argparse.Namespace, seed: Optional[int]) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    if spec.group == "eyeballing":
        kwargs.update(
            canvas_width=args.canvas_width,
            seed=seed,
            record_video=args.video,
        )
        if args.point_radius is not None:
            kwargs["point_radius"] = args.point_radius
        if args.line_width is not None:
            kwargs["line_width"] = args.line_width
    elif spec.name == "maze_square":
        kwargs.update(
            rows=args.maze_rows,
            cols=args.maze_cols,
            cell_size=args.maze_cell_size,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.name == "maze_hexagon":
        kwargs.update(
            radius=args.hex_radius,
            cell_radius=args.hex_cell_size,
            wall_thickness=args.hex_wall_thickness,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.name == "maze_labyrinth":
        kwargs.update(
            rings=args.lab_rings,
            segments=args.lab_segments,
            ring_width=args.lab_cell_size,
            wall_thickness=args.lab_wall_thickness,
            canvas_width=args.canvas_width,
            seed=seed,
            video=args.video,
        )
    elif spec.group == "visual_puzzle":
        kwargs.update(
            seed=seed,
            target_size=(512, 512),
            unique=True,
            video=args.video,
        )
    return kwargs


def _generate_visual_puzzle_worker(
    spec: TaskSpec,
    output_dir: Path,
    count: int,
    kwargs: Dict[str, Any],
) -> Dict[str, Any]:
    module = importlib.import_module(spec.module)
    pattern_class = load_generator_class(spec)

    pattern_kwargs = dict(kwargs)
    seed = pattern_kwargs.pop("seed", None)
    target_size = tuple(pattern_kwargs.pop("target_size", (512, 512)))
    unique = bool(pattern_kwargs.pop("unique", True))
    video = bool(pattern_kwargs.pop("video", False))

    if seed is not None:
        random.seed(seed)
        try:
            module.np.random.seed(seed)
        except Exception:
            pass

    pattern = pattern_class(**pattern_kwargs)
    module_root = Path(module.__file__).resolve().parent
    for attr_name in dir(pattern):
        if not attr_name.startswith("path_"):
            continue
        raw_path = getattr(pattern, attr_name, None)
        if not isinstance(raw_path, str) or not raw_path or Path(raw_path).is_absolute():
            continue
        setattr(pattern, attr_name, (module_root / raw_path).as_posix())

    output_dir.mkdir(parents=True, exist_ok=True)
    puzzles_dir = output_dir / "puzzles"
    solutions_dir = output_dir / "solutions"
    puzzles_dir.mkdir(parents=True, exist_ok=True)
    solutions_dir.mkdir(parents=True, exist_ok=True)

    samples: List[Dict[str, Any]] = []
    seen = set()
    question_idx = 0
    attempts = 0
    max_attempts = max(count * 100, 100)

    while len(samples) < count:
        attempts += 1
        if attempts > max_attempts:
            break

        sample, puzzle_image, solution_image = pattern.make_sample()
        if unique:
            image_string = module.convert_image_to_text(puzzle_image)
            if image_string in seen:
                continue
            seen.add(image_string)

        sample = dict(sample)
        sample["id"] = f"{spec.name}-{question_idx:02d}"
        instruction = module.pattern_instructions[spec.name]
        ti2v_prompt = _build_visual_puzzle_ti2v_prompt(sample, instruction, module.VIDEOGEN_INSTRUCTION_COMMON)
        ti2t_prompt = _build_visual_puzzle_ti2t_prompt(sample)
        sample["ti2v_prompt"] = ti2v_prompt
        sample["prompt"] = ti2v_prompt
        sample["ti2i_prompt"] = None
        sample["ti2t_prompt"] = ti2t_prompt
        sample["vlm_prompt"] = ti2t_prompt
        sample["ti2ti_prompt"] = None

        puzzle_image = module.pad_image(puzzle_image, target_size=target_size)
        solution_image = module.pad_image(solution_image, target_size=target_size)

        image_path = puzzles_dir / f"{question_idx:02d}.png"
        solution_path = solutions_dir / f"{question_idx:02d}.png"
        puzzle_image.save(image_path)
        solution_image.save(solution_path)
        solution_video_rel = None
        video_fps_val = None
        video_num_frames_val = None
        if video:
            video_file = solutions_dir / f"{question_idx:02d}.mp4"
            num_frames = module.save_visual_puzzle_video(
                puzzle_image, solution_image, str(video_file), fps=16,
            )
            if num_frames > 0 and video_file.exists():
                solution_video_rel = video_file.relative_to(output_dir).as_posix()
                video_fps_val = 16
                video_num_frames_val = num_frames

        sample["image"] = image_path.relative_to(output_dir).as_posix()
        sample["reasoning_image"] = sample["image"]
        sample["solution_image_path"] = solution_path.relative_to(output_dir).as_posix()
        sample["solution_video_path"] = solution_video_rel
        sample["video_fps"] = video_fps_val
        sample["video_num_frames"] = video_num_frames_val
        samples.append(sample)
        question_idx += 1

    metadata_path = output_dir / "data.json"
    write_json(metadata_path, samples)
    return {
        "task": spec.name,
        "worker_dir": output_dir.as_posix(),
        "count": len(samples),
        "metadata": metadata_path.as_posix(),
    }


def _prefixed_dest_name(worker_name: str, filename: str) -> str:
    return f"{worker_name}__{filename}"


def _move_asset_value(
    raw_value: Any,
    *,
    worker_dir: Path,
    task_dir: Path,
    worker_name: str,
) -> Any:
    if not isinstance(raw_value, str):
        return raw_value
    src = Path(raw_value)
    if not src.is_absolute():
        src = worker_dir / src
    if not src.exists():
        return raw_value

    rel_subdir = Path(raw_value).parent if not Path(raw_value).is_absolute() else Path(src.name).parent
    if str(rel_subdir) == ".":
        rel_subdir = Path("")
    dest_dir = task_dir / rel_subdir
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_name = _prefixed_dest_name(worker_name, src.name)
    dest = dest_dir / dest_name
    shutil.move(src.as_posix(), dest.as_posix())
    return dest.relative_to(task_dir).as_posix()


def _merge_worker_records(task_dir: Path, worker_dirs: Sequence[Path]) -> List[Dict[str, Any]]:
    merged: List[Dict[str, Any]] = []
    path_keys = [
        "image",
        "reasoning_image",
        "solution_image_path",
        "solution_video_path",
        "video",
        "puzzle",
        "solution",
    ]
    for worker_dir in worker_dirs:
        worker_meta = worker_dir / "data.json"
        if not worker_meta.exists():
            continue
        payload = read_json(worker_meta)
        if not isinstance(payload, list):
            continue
        worker_name = worker_dir.name
        for record in payload:
            if not isinstance(record, dict):
                continue
            record = dict(record)
            remapped_paths: Dict[str, Any] = {}
            for key in path_keys:
                if key in record:
                    raw_value = record[key]
                    if isinstance(raw_value, str) and raw_value in remapped_paths:
                        record[key] = remapped_paths[raw_value]
                    else:
                        moved_value = _move_asset_value(
                            raw_value,
                            worker_dir=worker_dir,
                            task_dir=task_dir,
                            worker_name=worker_name,
                        )
                        record[key] = moved_value
                        if isinstance(raw_value, str):
                            remapped_paths[raw_value] = moved_value
            if "images" in record and isinstance(record["images"], list):
                remapped_images: List[Any] = []
                for item in record["images"]:
                    if isinstance(item, str) and item in remapped_paths:
                        remapped_images.append(remapped_paths[item])
                        continue
                    moved_item = _move_asset_value(
                        item,
                        worker_dir=worker_dir,
                        task_dir=task_dir,
                        worker_name=worker_name,
                    )
                    remapped_images.append(moved_item)
                    if isinstance(item, str):
                        remapped_paths[item] = moved_item
                record["images"] = remapped_images
            merged.append(record)
    return merged


def _ensure_unique_record_ids(records: Sequence[Dict[str, Any]], task_name: str) -> List[Dict[str, Any]]:
    unique_records: List[Dict[str, Any]] = []
    seen_counts: Dict[str, int] = {}
    for index, record in enumerate(records):
        normalized = dict(record)
        base_id = str(normalized.get("id") or f"{task_name}-{index:05d}")
        duplicate_count = seen_counts.get(base_id, 0)
        seen_counts[base_id] = duplicate_count + 1
        normalized["id"] = base_id if duplicate_count == 0 else f"{base_id}__{duplicate_count:02d}"
        unique_records.append(normalized)
    return unique_records


def _generate_worker(job: Dict[str, Any]) -> Dict[str, Any]:
    spec: TaskSpec = job["spec"]
    output_dir: Path = Path(job["output_dir"])
    count: int = int(job["count"])
    kwargs: Dict[str, Any] = dict(job["kwargs"])

    if spec.group == "visual_puzzle":
        return _generate_visual_puzzle_worker(spec, output_dir, count, kwargs)

    output_dir.mkdir(parents=True, exist_ok=True)
    generator_class = load_generator_class(spec)
    signature = inspect.signature(generator_class.__init__)
    accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
    if not accepts_kwargs:
        allowed = {name for name in signature.parameters if name not in {"self", "output_dir"}}
        kwargs = {key: value for key, value in kwargs.items() if key in allowed}

    generator = generator_class(output_dir=output_dir, **kwargs)
    records = [generator.create_random_puzzle() for _ in range(count)]
    metadata_path = output_dir / "data.json"
    generator.write_metadata(records, metadata_path, append=False)

    return {
        "task": spec.name,
        "worker_dir": output_dir.as_posix(),
        "count": count,
        "metadata": metadata_path.as_posix(),
    }


def run_generation(args: argparse.Namespace) -> Dict[str, Any]:
    requested_tasks = resolve_requested_tasks(args.tasks, args.task_groups, args.exclude_tasks)
    if not requested_tasks:
        raise ValueError("No tasks selected. Check --tasks / --task_groups / --exclude_tasks")

    output_root = Path(args.output_root).expanduser().resolve()
    tasks_root = output_root / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)

    task_config = parse_task_config(args.task_config_path, args.task_config)

    manifest_samples = []
    report: Dict[str, Any] = {
        "output_root": output_root.as_posix(),
        "tasks": {},
        "total_generated_records": 0,
        "manifest_before_dedup": 0,
        "manifest_after_dedup": 0,
        "manifest_dropped_duplicates": 0,
    }

    for task in requested_tasks:
        spec = TASK_SPECS[task]
        missing = [req for req in spec.requires if not getattr(args, req, None)]
        if missing:
            report["tasks"][task] = {"status": "skipped", "reason": f"missing required args: {missing}"}
            continue

        task_dir = tasks_root / task
        task_dir.mkdir(parents=True, exist_ok=True)
        tmp_root = task_dir / ".tmp_workers"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        workers = max(1, min(args.num_workers, args.count))
        counts = split_counts(args.count, workers)

        jobs: List[Dict[str, Any]] = []
        for worker_idx, count in enumerate(counts):
            if count <= 0:
                continue
            worker_dir = tmp_root / f"worker_{worker_idx:02d}"
            base_seed = args.seed if args.seed is not None else None
            seed = (base_seed + worker_idx) if base_seed is not None else None
            kwargs = build_default_kwargs(spec, args, seed)
            override_kwargs = task_config.get(task, {})
            if override_kwargs:
                kwargs.update(override_kwargs)
            jobs.append(
                {
                    "spec": spec,
                    "output_dir": worker_dir,
                    "count": count,
                    "kwargs": kwargs,
                }
            )

        results: List[Dict[str, Any]] = []
        if len(jobs) <= 1:
            for job in jobs:
                results.append(_generate_worker(job))
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(processes=max(1, len(jobs))) as pool:
                for result in pool.imap_unordered(_generate_worker, jobs):
                    results.append(result)

        worker_dirs = [Path(item["worker_dir"]) for item in results]
        merged_records = _ensure_unique_record_ids(_merge_worker_records(task_dir, worker_dirs), task)

        merged_metadata = task_dir / "data.json"
        write_json(merged_metadata, merged_records)

        if tmp_root.exists():
            shutil.rmtree(tmp_root)

        task_samples = []
        for record in merged_records:
            sample = build_canonical_sample(
                record,
                task_dir=task_dir,
                metadata_path=merged_metadata,
                source_task_name=task,
            )
            if sample is not None:
                task_samples.append(sample)

        expected_records = args.count
        if len(merged_records) != expected_records:
            raise RuntimeError(
                f"Task '{task}' generated {len(merged_records)} records, expected {expected_records}. "
                "This usually indicates worker failure or premature termination.",
            )
        if len(task_samples) != expected_records:
            raise RuntimeError(
                f"Task '{task}' canonicalized {len(task_samples)} samples, expected {expected_records}. "
                "Check task metadata and generated assets for dropped records.",
            )

        manifest_samples.extend(task_samples)
        report["tasks"][task] = {
            "status": "ok",
            "workers": len(jobs),
            "records": len(merged_records),
            "canonical_samples": len(task_samples),
            "metadata_path": merged_metadata.as_posix(),
        }
        report["total_generated_records"] += len(merged_records)

    report["manifest_before_dedup"] = len(manifest_samples)
    deduped_samples, dropped = dedup_samples(manifest_samples)
    report["manifest_after_dedup"] = len(deduped_samples)
    report["manifest_dropped_duplicates"] = dropped

    manifest_path = output_root / "canonical_manifest.jsonl"
    write_manifest(manifest_path, deduped_samples)

    report_path = output_root / "generation_report.json"
    write_json(report_path, report)

    return {
        "manifest_path": manifest_path,
        "report_path": report_path,
        "report": report,
    }


def build_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("generate", help="Generate puzzle datasets and canonical manifest")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--tasks", nargs="+", default=["all"], help="Task names or all")
    parser.add_argument(
        "--task-groups",
        nargs="+",
        default=["eyeballing", "maze", "visual_puzzle"],
        choices=["eyeballing", "maze", "visual_puzzle"],
    )
    parser.add_argument("--exclude-tasks", nargs="+", default=[])
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--video", action="store_true", help="Generate solution videos when supported")
    parser.add_argument("--canvas-width", type=int, default=512)

    parser.add_argument("--point-radius", type=int, default=None)
    parser.add_argument("--line-width", type=int, default=None)

    parser.add_argument("--maze-rows", type=int, default=17)
    parser.add_argument("--maze-cols", type=int, default=17)
    parser.add_argument("--maze-cell-size", type=int, default=None)

    parser.add_argument("--hex-radius", type=int, default=3)
    parser.add_argument("--hex-cell-size", type=int, default=None)
    parser.add_argument("--hex-wall-thickness", type=int, default=None)

    parser.add_argument("--lab-rings", type=int, default=3)
    parser.add_argument("--lab-segments", type=int, default=8)
    parser.add_argument("--lab-cell-size", type=int, default=None)
    parser.add_argument("--lab-wall-thickness", type=int, default=None)

    parser.add_argument("--task-config-path", type=str, default=None)
    parser.add_argument("--task-config", type=str, default=None)

    parser.set_defaults(func=_cmd_generate)


def _cmd_generate(args: argparse.Namespace) -> None:
    result = run_generation(args)
    print("=" * 60)
    print("VideoThinkBench data generate finished")
    print(f"Manifest: {result['manifest_path']}")
    print(f"Report: {result['report_path']}")
    print("=" * 60)
