from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image


def _frame_to_rgb_array(frame: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.array(frame.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("Video frames must have shape HxWx3")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        else:
            array = array.copy()
    return np.ascontiguousarray(array)


def _prepare_frames(frames: Sequence[Image.Image | np.ndarray]) -> tuple[list[np.ndarray], int, int]:
    if not frames:
        return [], 0, 0

    arrays = [_frame_to_rgb_array(frame) for frame in frames]
    height, width = arrays[0].shape[:2]
    for array in arrays[1:]:
        if array.shape[:2] != (height, width):
            raise ValueError("All video frames must share the same size")

    target_width = width + (width % 2)
    target_height = height + (height % 2)
    if target_width != width or target_height != height:
        padded_arrays: list[np.ndarray] = []
        for array in arrays:
            padded = np.full((target_height, target_width, 3), 255, dtype=np.uint8)
            padded[:height, :width] = array
            padded_arrays.append(padded)
        arrays = padded_arrays
        width = target_width
        height = target_height

    return arrays, width, height


def encode_rgb_frames_to_mp4(
    frames: Sequence[Image.Image | np.ndarray],
    path: Path,
    *,
    fps: int = 16,
) -> bool:
    frame_arrays, width, height = _prepare_frames(frames)
    if not frame_arrays:
        return False
    if fps <= 0:
        raise ValueError("fps must be positive")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is not None:
        cmd = [
            ffmpeg_path,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{width}x{height}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-preset",
            "fast",
            "-crf",
            "23",
            str(path),
        ]
        try:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError as exc:
            print(f"Error: Failed to launch ffmpeg for {path}: {exc}", flush=True)
            return False

        stderr_output = b""
        try:
            for frame in frame_arrays:
                if proc.stdin is None:
                    raise BrokenPipeError("ffmpeg stdin pipe is unavailable")
                proc.stdin.write(frame.tobytes())
        except BrokenPipeError:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.stderr is not None:
                stderr_output = proc.stderr.read()
            proc.wait()
        else:
            if proc.stdin is not None:
                proc.stdin.close()
            if proc.stderr is not None:
                stderr_output = proc.stderr.read()
            proc.wait()
        finally:
            if proc.stderr is not None:
                proc.stderr.close()

        if proc.returncode == 0 and path.exists():
            return True

        error_message = stderr_output.decode("utf-8", errors="replace").strip()
        print(
            f"Error: ffmpeg failed to encode {path}: {error_message or f'return code {proc.returncode}'}",
            flush=True,
        )
        return False

    try:
        import cv2
    except ImportError:
        print(f"Error: ffmpeg is unavailable and OpenCV fallback could not be imported for {path}", flush=True)
        return False

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        writer.release()
        print(f"Error: Failed to open OpenCV avc1 writer for {path}", flush=True)
        return False

    for frame in frame_arrays:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return path.exists()
