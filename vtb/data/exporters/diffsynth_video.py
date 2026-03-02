from __future__ import annotations

from pathlib import Path
from typing import Sequence

from vtb.data.scan import filter_by_task_group
from vtb.schemas import CanonicalSample
from vtb.utils.io import write_csv


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
        rows.append(
            {
                "video": sample.assets.solution_video,
                "prompt": sample.prompt_train,
                "task_type": sample.task_type,
                "task_group": sample.task_group,
                "id": sample.id,
            }
        )

    write_csv(output_path, ["video", "prompt", "task_type", "task_group", "id"], rows)
    return output_path
