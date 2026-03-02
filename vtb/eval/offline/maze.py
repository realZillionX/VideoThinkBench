from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from vtb.schemas import CanonicalSample, EvalRecord
from vtb.eval.offline.common import resolve_candidate_image


def _maze_evaluator_class(task_type: str):
    if task_type.startswith("maze_hexagon"):
        from data.puzzle.maze.maze_hexagon.evaluator import MazeHexagonEvaluator as EvaluatorClass

        return EvaluatorClass
    if task_type.startswith("maze_labyrinth"):
        from data.puzzle.maze.maze_labyrinth.evaluator import MazeLabyrinthEvaluator as EvaluatorClass

        return EvaluatorClass
    from data.puzzle.maze.maze_square.evaluator import MazeEvaluator as EvaluatorClass

    return EvaluatorClass


def run_offline_maze(samples: Sequence[CanonicalSample], pred_root: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    evaluator_cache: Dict[Tuple[str, str], object] = {}

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
            cache_key = (sample.task_type, metadata_path.as_posix())
            evaluator = evaluator_cache.get(cache_key)
            if evaluator is None:
                evaluator_class = _maze_evaluator_class(sample.task_type)
                task_dir = sample.source.get("task_dir")
                base_dir = Path(task_dir).expanduser().resolve() if task_dir else metadata_path.parent
                evaluator = evaluator_class(metadata_path, base_dir=base_dir)
                evaluator_cache[cache_key] = evaluator

            result = evaluator.evaluate(sample.id, candidate_image)
            metrics = result.to_dict() if hasattr(result, "to_dict") else dict(result)
            offline_pass = bool(
                metrics.get("overlaps_walls") is False
                and metrics.get("touches_start") is True
                and metrics.get("touches_goal") is True
                and metrics.get("connected") is True
            )
            record.prediction_asset = candidate_image.as_posix()
            record.offline_metrics = metrics
            record.offline_pass = offline_pass
        except Exception as exc:  # pragma: no cover - guarded by integration tests
            record.error = str(exc)
            record.offline_pass = False
        records.append(record)

    return records
