from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CanonicalAssets:
    puzzle_image: str
    solution_image: str
    solution_video: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_image": self.puzzle_image,
            "solution_image": self.solution_image,
            "solution_video": self.solution_video,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalAssets":
        return CanonicalAssets(
            puzzle_image=str(payload["puzzle_image"]),
            solution_image=str(payload["solution_image"]),
            solution_video=(str(payload["solution_video"]) if payload.get("solution_video") else None),
            video_fps=int(payload["video_fps"]) if payload.get("video_fps") is not None else None,
            video_num_frames=int(payload["video_num_frames"]) if payload.get("video_num_frames") is not None else None,
        )


@dataclass
class CanonicalAnswer:
    path_cell_ids: Optional[List[int]] = None
    correct_option: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_cell_ids": self.path_cell_ids,
            "correct_option": self.correct_option,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalAnswer":
        raw_ids = payload.get("path_cell_ids")
        path_cell_ids: Optional[List[int]] = None
        if isinstance(raw_ids, list):
            path_cell_ids = [int(item) for item in raw_ids]
        raw_option = payload.get("correct_option")
        correct_option = str(raw_option) if raw_option is not None else None
        return CanonicalAnswer(path_cell_ids=path_cell_ids, correct_option=correct_option)


@dataclass
class CanonicalSample:
    id: str
    task_group: str
    task_type: str
    prompt_raw: str
    prompt_train: str
    assets: CanonicalAssets
    answer: CanonicalAnswer
    source: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_group": self.task_group,
            "task_type": self.task_type,
            "prompt_raw": self.prompt_raw,
            "prompt_train": self.prompt_train,
            "assets": self.assets.to_dict(),
            "answer": self.answer.to_dict(),
            "source": self.source,
            "extra": self.extra,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalSample":
        return CanonicalSample(
            id=str(payload["id"]),
            task_group=str(payload["task_group"]),
            task_type=str(payload["task_type"]),
            prompt_raw=str(payload.get("prompt_raw", "")),
            prompt_train=str(payload.get("prompt_train", "")),
            assets=CanonicalAssets.from_dict(dict(payload.get("assets") or {})),
            answer=CanonicalAnswer.from_dict(dict(payload.get("answer") or {})),
            source=dict(payload.get("source") or {}),
            extra=dict(payload.get("extra") or {}),
        )


@dataclass
class EvalRecord:
    sample_id: str
    task_group: str
    task_type: str
    input_asset: str
    prediction_asset: Optional[str] = None
    prediction_text: Optional[str] = None
    offline_metrics: Dict[str, Any] = field(default_factory=dict)
    offline_pass: Optional[bool] = None
    infer_meta: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "task_group": self.task_group,
            "task_type": self.task_type,
            "input_asset": self.input_asset,
            "prediction_asset": self.prediction_asset,
            "prediction_text": self.prediction_text,
            "offline_metrics": self.offline_metrics,
            "offline_pass": self.offline_pass,
            "infer_meta": self.infer_meta,
            "error": self.error,
        }
