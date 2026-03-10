from __future__ import annotations

from pathlib import Path
from typing import Sequence

from data.scan import filter_by_task_group
from core.schemas import CanonicalSample
from core.io import write_csv


def export_diffsynth_video(
    samples: Sequence[CanonicalSample],
    *,
    output_path: Path,
    task_groups: Sequence[str],
) -> Path:
    filtered = filter_by_task_group(samples, task_groups)
    rows = []
    for sample in filtered:
        if not sample.assets.solution_video:
            continue
        raw = sample.extra.get("raw_record") or {}
        vlm_answer = raw.get("vlm_answer")
        sol_image = raw.get("solution_image_path")
        ti2ti_text = str(vlm_answer) if vlm_answer else None
        ti2ti_image = str(sol_image) if sol_image else None
        rows.append(
            {
                "video": sample.assets.solution_video,
                "prompt": sample.prompt_train,
                "task_type": sample.task_type,
                "task_group": sample.task_group,
                "id": sample.id,
                "ti2ti_text": ti2ti_text or "",
                "ti2ti_image": ti2ti_image or "",
            }
        )

    write_csv(
        output_path,
        ["video", "prompt", "task_type", "task_group", "id", "ti2ti_text", "ti2ti_image"],
        rows,
    )
    return output_path
