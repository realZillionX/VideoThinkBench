from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

from vtb.utils.paths import first_existing_path


DEFAULT_CANDIDATE_NAMES = [
    "last_frame.png",
    "generated.png",
    "candidate.png",
    "output.png",
    "prediction.png",
]


def resolve_candidate_image(pred_root: Path, sample_id: str) -> Optional[Path]:
    sample_dir = pred_root / sample_id
    candidates = []
    if sample_dir.exists() and sample_dir.is_dir():
        candidates.extend(sample_dir / name for name in DEFAULT_CANDIDATE_NAMES)
        candidates.extend(sample_dir.glob("*.png"))
        candidates.extend(sample_dir.glob("*.jpg"))
    candidates.extend(
        pred_root / f"{sample_id}{ext}" for ext in [".png", ".jpg", ".jpeg", ".webp"]
    )
    return first_existing_path(candidates)


def resolve_attempt_dir(candidate_image: Path) -> Path:
    return candidate_image.parent
