from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

from core.io import read_jsonl
from core.paths import resolve_dataset_asset_path


def load_vlm_rows(dataset_path: Path) -> List[Dict]:
    return read_jsonl(dataset_path)


def load_image_rows(dataset_path: Path) -> List[Dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Image dataset must be a list: {dataset_path}")
    return [row for row in payload if isinstance(row, dict)]


def load_video_rows(dataset_path: Path) -> List[Dict]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def resolve_video_input(
    row: Dict,
    *,
    dataset_path: Path,
    dataset_root: Path | None,
) -> Path:
    raw_video = str(row.get("video") or "").strip()
    if not raw_video:
        raise ValueError("CSV row missing 'video' field")
    return resolve_dataset_asset_path(raw_video, dataset_root=dataset_root, dataset_file=dataset_path)
