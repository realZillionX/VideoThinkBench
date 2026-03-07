from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from PIL import Image

from evaluation.infer.common import load_video_rows, resolve_video_input
from evaluation.infer.common_diffsynth import ensure_diffsynth_path, load_wan_video_pipeline
from core.schemas import EvalRecord


def _load_first_frame(video_path: Path, width: int, height: int) -> Image.Image:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("opencv-python is required for video inference") from exc

    capture = cv2.VideoCapture(video_path.as_posix())
    ok, frame = capture.read()
    capture.release()
    if not ok or frame is None:
        raise RuntimeError(f"Failed to read first frame from {video_path}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame)
    return image.resize((width, height), resample=Image.NEAREST)


def run_video_infer(
    *,
    dataset_path: Path,
    model_path: str,
    output_dir: Path,
    mode: str,
    lora_path: Optional[str],
    num_samples: Optional[int],
    diffsynth_path: Optional[str],
    dataset_root: Optional[Path],
    width: int = 480,
    height: int = 896,
    num_frames: int = 81,
    seed: int = 42,
    device: str = "cuda",
) -> List[EvalRecord]:
    ensure_diffsynth_path(diffsynth_path)

    try:
        from diffsynth.core.data import save_video
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("DiffSynth-Studio is required for video inference") from exc

    pipe = load_wan_video_pipeline(model_base_path=model_path, lora_ckpt=lora_path, device=device)
    rows = load_video_rows(dataset_path)
    if num_samples is not None and num_samples > 0:
        rows = rows[:num_samples]

    samples_dir = output_dir / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    records: List[EvalRecord] = []
    for index, row in enumerate(rows):
        sample_id = str(row.get("id") or Path(str(row.get("video") or f"row_{index:06d}")).stem)
        task_group = str(row.get("task_group") or "unknown")
        task_type = str(row.get("task_type") or "unknown")
        prompt = str(row.get("prompt") or "")

        result = EvalRecord(
            sample_id=sample_id,
            task_group=task_group,
            task_type=task_type,
            input_asset=str(row.get("video") or ""),
        )
        try:
            source_video = resolve_video_input(row, dataset_path=dataset_path, dataset_root=dataset_root)
            if not source_video.exists():
                raise FileNotFoundError(f"Source video not found: {source_video}")

            input_image = _load_first_frame(source_video, width, height)
            video = pipe(
                prompt=prompt,
                negative_prompt="",
                input_image=input_image,
                num_frames=num_frames,
                height=height,
                width=width,
                seed=seed,
                tiled=True,
            )

            sample_dir = samples_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            video_out = sample_dir / "generated.mp4"
            save_video(video, video_out.as_posix(), fps=15, quality=5)
            result.prediction_asset = video_out.as_posix()

            last_frame = sample_dir / "last_frame.png"
            if isinstance(video, (list, tuple)) and video:
                video[-1].save(last_frame)

            result.infer_meta = {
                "mode": mode,
                "prompt": prompt,
                "source_video": source_video.as_posix(),
                "last_frame": last_frame.as_posix() if last_frame.exists() else None,
            }
            result.input_asset = source_video.as_posix()
        except Exception as exc:  # pragma: no cover
            result.error = str(exc)
        records.append(result)

    return records
