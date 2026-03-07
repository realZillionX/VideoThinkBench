from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional


def to_absolute(path_value: str, base_dir: Path) -> Path:
    candidate = Path(path_value)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def resolve_dataset_asset_path(
    raw_path: str,
    *,
    dataset_root: Optional[Path],
    dataset_file: Optional[Path],
) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    if dataset_root is not None:
        root_candidate = (dataset_root / candidate).resolve()
        if root_candidate.exists():
            return root_candidate

    if dataset_file is not None:
        file_candidate = (dataset_file.parent / candidate).resolve()
        if file_candidate.exists():
            return file_candidate

    if dataset_root is not None:
        return (dataset_root / candidate).resolve()
    if dataset_file is not None:
        return (dataset_file.parent / candidate).resolve()
    return candidate.resolve()


def first_existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None
