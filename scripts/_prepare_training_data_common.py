from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if REPO_ROOT.as_posix() not in sys.path:
    sys.path.insert(0, REPO_ROOT.as_posix())

from data.scan import build_samples_from_data_root, write_manifest  # noqa: E402


TASK_GROUP_CHOICES = ("eyeballing", "maze", "visual_puzzle")


def normalize_task_groups(task_groups: Sequence[str] | None) -> list[str]:
    if not task_groups:
        return list(TASK_GROUP_CHOICES)
    normalized = []
    for task_group in task_groups:
        task_group_clean = str(task_group).strip()
        if task_group_clean not in TASK_GROUP_CHOICES:
            raise ValueError(f"Unsupported task group: {task_group_clean}")
        normalized.append(task_group_clean)
    return normalized


def load_samples(data_root: str, task_groups: Sequence[str] | None) -> tuple[Path, list[str], list]:
    dataset_root = Path(data_root).expanduser().resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    normalized_task_groups = normalize_task_groups(task_groups)
    samples = build_samples_from_data_root(dataset_root, task_groups=normalized_task_groups)
    return dataset_root, normalized_task_groups, samples


def maybe_write_manifest(manifest_path: str | None, samples: Sequence) -> Path | None:
    if not manifest_path:
        return None
    output_path = Path(manifest_path).expanduser().resolve()
    write_manifest(output_path, samples)
    return output_path
