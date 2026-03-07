from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from data.scan import filter_by_task_group
from core.schemas import CanonicalSample
from core.io import write_json


def export_diffsynth_image(
    samples: Sequence[CanonicalSample],
    *,
    output_path: Path,
    task_groups: Sequence[str],
) -> Path:
    filtered = filter_by_task_group(samples, task_groups)
    rows = []
    for sample in filtered:
        rows.append(
            {
                "id": sample.id,
                "prompt": sample.prompt_train,
                "image": sample.assets.solution_image,
                "edit_image": sample.assets.puzzle_image,
                "task_type": sample.task_type,
                "task_group": sample.task_group,
            }
        )
    write_json(output_path, rows)
    return output_path
