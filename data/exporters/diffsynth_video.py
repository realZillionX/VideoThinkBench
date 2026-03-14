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
        prompt = sample.prompt_for("ti2v")
        if not sample.assets.solution_video or not prompt:
            continue
        raw = sample.extra.get("raw_record") or {}
        ti2ti_answer = raw.get("ti2ti_answer") if isinstance(raw, dict) else None
        ti2ti_enabled = bool(sample.prompt_for("ti2ti"))
        ti2ti_text = None
        if ti2ti_enabled and isinstance(ti2ti_answer, dict):
            text_value = str(ti2ti_answer.get("text") or "").strip()
            ti2ti_text = text_value or None
        if ti2ti_enabled and ti2ti_text is None:
            fallback_text = str(raw.get("ti2t_answer") or raw.get("vlm_answer") or "").strip()
            ti2ti_text = fallback_text or None
        ti2ti_image = sample.assets.solution_image if ti2ti_enabled else None
        rows.append(
            {
                "video": sample.assets.solution_video,
                "prompt": prompt,
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
