from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List, Sequence

from data.scan import filter_by_task_group
from core.schemas import CanonicalSample
from core.io import write_jsonl
from core.prompts import build_vlm_user_prompt


def _solution_text(sample: CanonicalSample) -> str:
    if sample.task_group == "maze":
        return json.dumps(sample.answer.path_cell_ids or [], ensure_ascii=False)
    solution = str(sample.answer.correct_option or "").strip()
    if sample.task_group == "eyeballing":
        return solution.upper()
    return solution


def _build_entry(sample: CanonicalSample, mode: str) -> dict:
    solution = _solution_text(sample)
    entry = {
        "id": sample.id,
        "messages": [{"role": "user", "content": build_vlm_user_prompt(sample.prompt_train, mode)}],
        "images": [sample.assets.puzzle_image],
        "solution": solution,
        "task_type": sample.task_type,
        "task_group": sample.task_group,
    }
    if mode == "sft":
        entry["messages"].append({"role": "assistant", "content": solution})
    return entry


def export_ms_swift(
    samples: Sequence[CanonicalSample],
    *,
    output_dir: Path,
    modes: Iterable[str],
    task_groups: Sequence[str],
) -> List[Path]:
    filtered = filter_by_task_group(samples, task_groups)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[Path] = []
    for mode in modes:
        mode_clean = mode.strip().lower()
        if mode_clean not in {"sft", "grpo"}:
            continue
        rows = [_build_entry(sample, mode_clean) for sample in filtered]
        out_path = output_dir / ("train_sft.jsonl" if mode_clean == "sft" else "train_grpo.jsonl")
        write_jsonl(out_path, rows)
        outputs.append(out_path)
    return outputs
