from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from vtb.eval.offline.common import resolve_candidate_image
from vtb.schemas import CanonicalSample, EvalRecord
from vtb.utils.nato import extract_first_nato_letter


def _normalize_option(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if len(text) == 1 and text.isalpha():
        return text.upper()
    return extract_first_nato_letter(text)


def _aggregate_option(metrics: Dict[str, Optional[str]]) -> Optional[str]:
    vote_order = ["image_option", "video_option", "text_option", "transcribe_option"]
    normalized = {key: _normalize_option(metrics.get(key)) for key in vote_order}
    votes = [normalized[key] for key in vote_order if normalized[key]]
    if not votes:
        return None

    counter = Counter(votes)
    best_count = max(counter.values())
    best = {label for label, count in counter.items() if count == best_count}
    if len(best) == 1:
        return next(iter(best))

    for key in vote_order:
        value = normalized.get(key)
        if value in best:
            return value
    return None


def run_offline_eyeballing(samples: Sequence[CanonicalSample], pred_root: Path) -> List[EvalRecord]:
    try:
        from data.puzzle.point_target_base import PointTargetPuzzleEvaluator
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Point-target evaluator dependencies are missing.") from exc

    records: List[EvalRecord] = []
    evaluator_cache: Dict[str, object] = {}

    for sample in samples:
        record = EvalRecord(
            sample_id=sample.id,
            task_group=sample.task_group,
            task_type=sample.task_type,
            input_asset=sample.assets.puzzle_image,
        )
        try:
            candidate_image = resolve_candidate_image(pred_root, sample.id)
            if candidate_image is None:
                raise FileNotFoundError(f"No prediction image found for sample_id={sample.id}")

            metadata_path = Path(sample.source.get("metadata_path", "")).expanduser().resolve()
            if not metadata_path.exists():
                raise FileNotFoundError(f"Metadata not found: {metadata_path}")

            evaluator = evaluator_cache.get(metadata_path.as_posix())
            if evaluator is None:
                task_dir = sample.source.get("task_dir")
                base_dir = Path(task_dir).expanduser().resolve() if task_dir else metadata_path.parent
                evaluator = PointTargetPuzzleEvaluator(metadata_path, base_dir=base_dir)
                evaluator_cache[metadata_path.as_posix()] = evaluator

            try:
                result = evaluator.evaluate(sample.id, candidate_image)
                metrics = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            except FileNotFoundError:
                puzzle = evaluator.get_record(sample.id)
                image_option, red_pixel_count, red_centroid = evaluator.image_option_from_path(candidate_image, puzzle)
                metrics = {
                    "puzzle_id": sample.id,
                    "correct_option": str(puzzle.get("correct_option", "")).strip().upper(),
                    "transcribe_option": None,
                    "video_option": None,
                    "image_option": image_option,
                    "text_option": None,
                    "red_pixel_count": red_pixel_count,
                    "red_centroid": red_centroid,
                    "attempt_dir": candidate_image.parent.as_posix(),
                }
            aggregated = _aggregate_option(metrics)
            correct = _normalize_option(metrics.get("correct_option"))
            offline_pass = bool(aggregated and correct and aggregated == correct)

            metrics["aggregated_option"] = aggregated
            metrics["normalized_correct_option"] = correct

            record.prediction_asset = candidate_image.as_posix()
            record.prediction_text = aggregated
            record.offline_metrics = metrics
            record.offline_pass = offline_pass
        except Exception as exc:  # pragma: no cover - guarded by integration tests
            record.error = str(exc)
            record.offline_pass = False
        records.append(record)

    return records
