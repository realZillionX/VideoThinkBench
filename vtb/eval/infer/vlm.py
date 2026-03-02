from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from vtb.eval.infer.common import load_vlm_rows
from vtb.schemas import EvalRecord


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def _extract_query_and_images(sample: dict) -> tuple[str, list]:
    messages = sample.get("messages") or []
    if messages and isinstance(messages, list) and isinstance(messages[0], dict):
        return str(messages[0].get("content") or ""), list(sample.get("images") or [])
    return str(sample.get("query") or ""), list(sample.get("images") or [])


def _score_prediction(prediction: str, solution: str) -> Optional[float]:
    try:
        from training.vlm.rewards.vlm_rewards import reward_eyeballing, reward_maze
    except Exception:
        return None

    sol = solution.strip()
    if not sol:
        return None
    try:
        if sol.startswith("[") and sol.endswith("]"):
            return float(reward_maze([prediction], [sol])[0])
        return float(reward_eyeballing([prediction], [sol])[0])
    except Exception:
        return None


def run_vlm_infer(
    *,
    dataset_path: Path,
    model_path: str,
    output_dir: Path,
    device: str,
    mode: str,
    num_samples: Optional[int] = None,
) -> List[EvalRecord]:
    try:
        from swift.llm import get_model_tokenizer, get_template, inference
        from swift.utils import seed_everything
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ms-swift is required for VLM inference") from exc

    rows = load_vlm_rows(dataset_path)
    if num_samples is not None and num_samples > 0:
        rows = rows[:num_samples]

    model, tokenizer = get_model_tokenizer(model_path, model_kwargs={"device_map": device})
    template = get_template(model.model_meta.template, tokenizer)
    seed_everything(42)

    records: List[EvalRecord] = []
    for row in rows:
        sample_id = str(row.get("id") or row.get("sample_id") or f"row_{len(records):06d}")
        task_group = str(row.get("task_group") or "unknown")
        task_type = str(row.get("task_type") or "unknown")
        query, images = _extract_query_and_images(row)
        result = EvalRecord(
            sample_id=sample_id,
            task_group=task_group,
            task_type=task_type,
            input_asset=(images[0] if images else ""),
        )
        try:
            prediction, _ = inference(model, template, query, images=images)
            result.prediction_text = prediction
            result.infer_meta = {
                "mode": mode,
                "query": query,
                "images": images,
            }
            solution = str(row.get("solution") or "")
            score = _score_prediction(prediction, solution)
            if score is not None:
                result.offline_metrics = {"score": score}
                if mode == "validate":
                    result.offline_pass = score >= 0.999
        except Exception as exc:  # pragma: no cover
            result.error = str(exc)
        records.append(result)

    return records
