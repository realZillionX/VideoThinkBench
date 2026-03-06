from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

from evaluators.offline.common import resolve_candidate_image
from utils.schemas import CanonicalSample, EvalRecord
from utils.paths import first_existing_path


DEFAULT_CANDIDATE_VIDEOS = [
    "generated.mp4",
    "output.mp4",
    "prediction.mp4",
    "result.mp4",
]


def _resolve_candidate_video(pred_root: Path, sample_id: str) -> Optional[Path]:
    sample_dir = pred_root / sample_id
    candidates = []
    if sample_dir.exists() and sample_dir.is_dir():
        candidates.extend(sample_dir / name for name in DEFAULT_CANDIDATE_VIDEOS)
        candidates.extend(sample_dir.glob("*.mp4"))
        candidates.extend(sample_dir.glob("*.mov"))
        candidates.extend(sample_dir.glob("*.webm"))
        candidates.extend(sample_dir.glob("*.avi"))
    return first_existing_path(candidates)


def _calculate_image_difference(candidate_image: Path, solution_image: Path) -> float:
    try:
        import cv2
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("opencv-python is required for visual puzzle offline evaluation.") from exc

    from evaluators.frame_matching.find_best_frame import calculate_difference

    candidate = cv2.imread(candidate_image.as_posix())
    solution = cv2.imread(solution_image.as_posix())
    if candidate is None:
        raise RuntimeError(f"Failed to read candidate image: {candidate_image}")
    if solution is None:
        raise RuntimeError(f"Failed to read solution image: {solution_image}")
    if candidate.shape != solution.shape:
        candidate = cv2.resize(candidate, (solution.shape[1], solution.shape[0]))
    return calculate_difference(candidate, solution, metric="euclidean")


def run_offline_visual_puzzle(
    samples: Sequence[CanonicalSample],
    pred_root: Path,
) -> List[EvalRecord]:
    from evaluators.frame_matching.find_best_frame import process_video

    records: List[EvalRecord] = []

    for sample in samples:
        record = EvalRecord(
            sample_id=sample.id,
            task_group=sample.task_group,
            task_type=sample.task_type,
            input_asset=sample.assets.puzzle_image,
        )
        try:
            solution_image = Path(sample.assets.solution_image).expanduser().resolve()
            if not solution_image.exists():
                raise FileNotFoundError(f"Solution image not found: {solution_image}")

            candidate_video = _resolve_candidate_video(pred_root, sample.id)
            candidate_image = resolve_candidate_image(pred_root, sample.id)
            difference: Optional[float] = None

            if candidate_video is not None:
                best_frame_path = candidate_video.parent / "best_frame.png"
                difference = process_video(
                    video_path=candidate_video,
                    solution_image_path=solution_image,
                    best_frame_path=best_frame_path,
                    frame_rate=1,
                    metric="euclidean",
                    compare_window=None,
                    binarization_threshold=245,
                )
                if best_frame_path.exists():
                    candidate_image = best_frame_path

            if candidate_image is None:
                raise FileNotFoundError(f"No prediction image or video found for sample_id={sample.id}")

            if difference is None:
                difference = _calculate_image_difference(candidate_image, solution_image)

            record.prediction_asset = candidate_image.as_posix()
            record.offline_metrics = {
                "difference": difference,
                "metric": "euclidean",
                "solution_image": solution_image.as_posix(),
                "candidate_image": candidate_image.as_posix(),
                "candidate_video": candidate_video.as_posix() if candidate_video else None,
            }
            record.offline_pass = None
        except Exception as exc:  # pragma: no cover - guarded by integration tests
            record.error = str(exc)
        records.append(record)

    return records
