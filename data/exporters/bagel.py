from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

from core.io import write_json, write_jsonl
from core.prompts import build_vlm_user_prompt
from core.schemas import CanonicalSample
from data.scan import filter_by_task_group


EDIT_DATASET_NAME = "videothinkbench_edit"
VLM_DATASET_NAME = "videothinkbench_vlm"


def _solution_text(sample: CanonicalSample) -> str:
    if sample.task_group == "maze":
        return json.dumps(sample.answer.path_cell_ids or [], ensure_ascii=False)
    solution = str(sample.answer.correct_option or "").strip()
    if sample.task_group == "eyeballing":
        return solution.upper()
    return solution


def _normalize_modes(modes: Iterable[str]) -> list[str]:
    normalized: list[str] = []
    for mode in modes:
        mode_clean = str(mode).strip().lower()
        if mode_clean in {"edit", "vlm"} and mode_clean not in normalized:
            normalized.append(mode_clean)
    if not normalized:
        raise ValueError("Bagel export requires at least one mode from: edit, vlm")
    return normalized


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_yaml_configs(
    output_dir: Path,
    *,
    include_edit: bool,
    include_vlm: bool,
    edit_num_files: int,
    vlm_num_samples: int,
) -> dict[str, Path]:
    configs: dict[str, Path] = {}
    sections: dict[str, str] = {}

    if include_edit:
        sections["edit"] = (
            "unified_edit:\n"
            "  dataset_names:\n"
            f"  - {EDIT_DATASET_NAME}\n"
            "  image_transform_args:\n"
            "    image_stride: 16\n"
            "    max_image_size: 1024\n"
            "    min_image_size: 512\n"
            "  vit_image_transform_args:\n"
            "    image_stride: 14\n"
            "    max_image_size: 518\n"
            "    min_image_size: 224\n"
            "  is_mandatory: false\n"
            "  num_used_data:\n"
            f"  - {edit_num_files}\n"
            "  weight: 1\n"
        )
        edit_path = output_dir / "config_edit.yaml"
        _write_text(edit_path, sections["edit"])
        configs["edit"] = edit_path

    if include_vlm:
        sections["vlm"] = (
            "vlm_sft:\n"
            "  dataset_names:\n"
            f"  - {VLM_DATASET_NAME}\n"
            "  image_transform_args:\n"
            "    image_stride: 14\n"
            "    max_image_size: 980\n"
            "    min_image_size: 378\n"
            "    max_pixels: 2007040\n"
            "  is_mandatory: true\n"
            "  shuffle_lines: true\n"
            "  shuffle_seed: 42\n"
            "  num_used_data:\n"
            f"  - {vlm_num_samples}\n"
            "  weight: 1\n"
        )
        vlm_path = output_dir / "config_vlm.yaml"
        _write_text(vlm_path, sections["vlm"])
        configs["vlm"] = vlm_path

    if include_edit and include_vlm:
        unified_path = output_dir / "config_unified.yaml"
        _write_text(unified_path, f"{sections['edit']}\n{sections['vlm']}")
        configs["unified"] = unified_path

    return configs


def _write_edit_dataset(
    samples: Sequence[CanonicalSample],
    *,
    output_dir: Path,
    rows_per_file: int,
) -> dict[str, object]:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("pyarrow is required for Bagel edit export") from exc

    parquet_dir = output_dir / "editing" / "parquet"
    parquet_dir.mkdir(parents=True, exist_ok=True)
    parquet_info_dir = output_dir / "editing" / "parquet_info"
    parquet_info_dir.mkdir(parents=True, exist_ok=True)

    shards: list[dict[str, object]] = []
    parquet_info: dict[str, dict[str, int]] = {}

    chunk_rows: list[CanonicalSample] = []
    shard_index = 0

    def flush() -> None:
        nonlocal shard_index
        if not chunk_rows:
            return

        image_lists = []
        instruction_lists = []
        sample_ids = []
        task_groups = []
        task_types = []
        answers = []
        for sample in chunk_rows:
            puzzle_bytes = Path(sample.assets.puzzle_image).read_bytes()
            solution_bytes = Path(sample.assets.solution_image).read_bytes()
            image_lists.append([puzzle_bytes, solution_bytes])
            instruction_lists.append([[sample.prompt_train]])
            sample_ids.append(sample.id)
            task_groups.append(sample.task_group)
            task_types.append(sample.task_type)
            answers.append(_solution_text(sample))

        shard_path = parquet_dir / f"{EDIT_DATASET_NAME}-{shard_index:05d}.parquet"
        table = pa.table(
            {
                "sample_id": pa.array(sample_ids, type=pa.string()),
                "task_group": pa.array(task_groups, type=pa.string()),
                "task_type": pa.array(task_types, type=pa.string()),
                "answer_text": pa.array(answers, type=pa.string()),
                "image_list": pa.array(image_lists, type=pa.list_(pa.binary())),
                "instruction_list": pa.array(instruction_lists, type=pa.list_(pa.list_(pa.string()))),
            }
        )
        pq.write_table(table, shard_path.as_posix(), row_group_size=len(chunk_rows))
        num_row_groups = pq.ParquetFile(shard_path.as_posix()).num_row_groups
        parquet_info[shard_path.as_posix()] = {"num_row_groups": int(num_row_groups)}
        shards.append(
            {
                "path": shard_path.as_posix(),
                "rows": len(chunk_rows),
                "num_row_groups": int(num_row_groups),
            }
        )
        chunk_rows.clear()
        shard_index += 1

    for sample in samples:
        chunk_rows.append(sample)
        if len(chunk_rows) >= rows_per_file:
            flush()
    flush()

    parquet_info_path = parquet_info_dir / f"{EDIT_DATASET_NAME}.json"
    write_json(parquet_info_path, parquet_info)

    return {
        "path": parquet_dir.as_posix(),
        "parquet_info_path": parquet_info_path.as_posix(),
        "num_files": len(shards),
        "num_samples": len(samples),
        "shards": shards,
    }


def _write_vlm_dataset(samples: Sequence[CanonicalSample], *, output_dir: Path) -> dict[str, object]:
    vlm_dir = output_dir / "vlm"
    vlm_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = vlm_dir / "train_sft.jsonl"

    rows = []
    for sample in samples:
        rows.append(
            {
                "id": sample.id,
                "image": sample.assets.puzzle_image,
                "task_group": sample.task_group,
                "task_type": sample.task_type,
                "solution": _solution_text(sample),
                "conversations": [
                    {
                        "from": "human",
                        "value": build_vlm_user_prompt(sample.prompt_train, "sft"),
                    },
                    {
                        "from": "gpt",
                        "value": _solution_text(sample),
                    },
                ],
            }
        )

    write_jsonl(jsonl_path, rows)
    return {
        "path": jsonl_path.as_posix(),
        "num_samples": len(rows),
    }


def export_bagel(
    samples: Sequence[CanonicalSample],
    *,
    output_dir: Path,
    task_groups: Sequence[str],
    modes: Iterable[str],
    parquet_rows_per_file: int = 1024,
) -> dict[str, Path]:
    filtered = filter_by_task_group(samples, task_groups)
    if not filtered:
        raise ValueError("No samples matched the requested task groups for Bagel export")

    selected_modes = _normalize_modes(modes)
    output_dir.mkdir(parents=True, exist_ok=True)

    include_edit = "edit" in selected_modes
    include_vlm = "vlm" in selected_modes

    dataset_info: dict[str, dict[str, dict[str, object]]] = {}
    summary: dict[str, object] = {
        "task_groups": list(task_groups),
        "modes": selected_modes,
        "num_samples": len(filtered),
    }
    outputs: dict[str, Path] = {}

    edit_info: dict[str, object] | None = None
    if include_edit:
        edit_info = _write_edit_dataset(
            filtered,
            output_dir=output_dir,
            rows_per_file=max(1, int(parquet_rows_per_file)),
        )
        dataset_info["unified_edit"] = {
            EDIT_DATASET_NAME: {
                "data_dir": str(edit_info["path"]),
                "num_files": int(edit_info["num_files"]),
                "num_total_samples": int(edit_info["num_samples"]),
                "parquet_info_path": str(edit_info["parquet_info_path"]),
            }
        }
        summary["edit"] = edit_info
        outputs["edit_parquet_dir"] = Path(str(edit_info["path"]))
        outputs["edit_parquet_info"] = Path(str(edit_info["parquet_info_path"]))

    vlm_info: dict[str, object] | None = None
    if include_vlm:
        vlm_info = _write_vlm_dataset(filtered, output_dir=output_dir)
        dataset_info["vlm_sft"] = {
            VLM_DATASET_NAME: {
                "data_dir": (output_dir / "vlm").as_posix(),
                "jsonl_path": str(vlm_info["path"]),
                "num_total_samples": int(vlm_info["num_samples"]),
            }
        }
        summary["vlm"] = vlm_info
        outputs["vlm_jsonl"] = Path(str(vlm_info["path"]))

    dataset_info_path = output_dir / "dataset_info.json"
    write_json(dataset_info_path, dataset_info)
    outputs["dataset_info"] = dataset_info_path

    config_paths = _write_yaml_configs(
        output_dir,
        include_edit=include_edit,
        include_vlm=include_vlm,
        edit_num_files=int(edit_info["num_files"]) if edit_info is not None else 0,
        vlm_num_samples=int(vlm_info["num_samples"]) if vlm_info is not None else 0,
    )
    outputs.update({f"config_{name}": path for name, path in config_paths.items()})

    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    outputs["summary"] = summary_path

    return outputs
