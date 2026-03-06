from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from evaluators.infer.common import load_image_rows
from evaluators.infer.common_diffsynth import ensure_diffsynth_path, load_qwen_image_pipeline
from utils.schemas import EvalRecord


def run_image_infer(
    *,
    dataset_path: Path,
    model_path: str,
    output_dir: Path,
    mode: str,
    lora_path: Optional[str],
    num_samples: Optional[int],
    diffsynth_path: Optional[str],
    seed: int = 42,
) -> List[EvalRecord]:
    ensure_diffsynth_path(diffsynth_path)
    pipe = load_qwen_image_pipeline(model_path=model_path, lora_path=lora_path)

    rows = load_image_rows(dataset_path)
    if num_samples is not None and num_samples > 0:
        rows = rows[:num_samples]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    records: List[EvalRecord] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or f"row_{index:06d}")
        task_group = str(row.get("task_group") or "unknown")
        task_type = str(row.get("task_type") or "unknown")
        prompt = str(row.get("prompt") or "")
        puzzle_image_path = Path(str(row.get("edit_image") or "")).expanduser().resolve()
        target_image_path = Path(str(row.get("image") or "")).expanduser().resolve()

        result = EvalRecord(
            sample_id=sample_id,
            task_group=task_group,
            task_type=task_type,
            input_asset=puzzle_image_path.as_posix(),
        )
        try:
            puzzle_image = Image.open(puzzle_image_path).convert("RGB")
            width, height = puzzle_image.size
            generated_image = pipe(
                prompt=prompt,
                edit_image=[puzzle_image],
                seed=seed,
                num_inference_steps=40,
                height=height,
                width=width,
                edit_image_auto_resize=True,
                zero_cond_t=True,
            )
            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            out_path = sample_dir / "generated.png"
            generated_image.save(out_path)
            result.prediction_asset = out_path.as_posix()
            result.infer_meta = {
                "mode": mode,
                "prompt": prompt,
                "target_image": target_image_path.as_posix(),
            }
        except Exception as exc:  # pragma: no cover
            result.error = str(exc)
        records.append(result)
    return records
