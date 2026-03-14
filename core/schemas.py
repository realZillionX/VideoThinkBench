from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class CanonicalAssets:
    puzzle_image: str
    solution_image: str
    reasoning_image: Optional[str] = None
    solution_video: Optional[str] = None
    video_fps: Optional[int] = None
    video_num_frames: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "puzzle_image": self.puzzle_image,
            "reasoning_image": self.reasoning_image,
            "solution_image": self.solution_image,
            "solution_video": self.solution_video,
            "video_fps": self.video_fps,
            "video_num_frames": self.video_num_frames,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalAssets":
        return CanonicalAssets(
            puzzle_image=str(payload["puzzle_image"]),
            reasoning_image=(str(payload["reasoning_image"]) if payload.get("reasoning_image") else None),
            solution_image=str(payload["solution_image"]),
            solution_video=(str(payload["solution_video"]) if payload.get("solution_video") else None),
            video_fps=int(payload["video_fps"]) if payload.get("video_fps") is not None else None,
            video_num_frames=int(payload["video_num_frames"]) if payload.get("video_num_frames") is not None else None,
        )

    def image_for_reasoning(self) -> str:
        return self.reasoning_image or self.puzzle_image


@dataclass
class CanonicalPrompts:
    ti2v: Optional[str] = None
    ti2i: Optional[str] = None
    ti2t: Optional[str] = None
    ti2ti: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ti2v": self.ti2v,
            "ti2i": self.ti2i,
            "ti2t": self.ti2t,
            "ti2ti": self.ti2ti,
        }

    @staticmethod
    def from_dict(payload: Dict[str, Any]) -> "CanonicalPrompts":
        return CanonicalPrompts(
            ti2v=(str(payload["ti2v"]) if payload.get("ti2v") else None),
            ti2i=(str(payload["ti2i"]) if payload.get("ti2i") else None),
            ti2t=(str(payload["ti2t"]) if payload.get("ti2t") else None),
            ti2ti=(str(payload["ti2ti"]) if payload.get("ti2ti") else None),
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
    prompts: CanonicalPrompts
    prompt_raw: str
    prompt_train: str
    assets: CanonicalAssets
    answer: CanonicalAnswer
    source: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    def _legacy_prompt(self, *keys: str) -> Optional[str]:
        raw_record = self.extra.get("raw_record") or {}
        if not isinstance(raw_record, dict):
            return None
        for key in keys:
            value = raw_record.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return None

    def _legacy_prompt_disabled(self, key: str) -> bool:
        raw_record = self.extra.get("raw_record") or {}
        if not isinstance(raw_record, dict) or key not in raw_record:
            return False
        value = raw_record.get(key)
        if value is None:
            return True
        return not str(value).strip()

    def prompt_for(self, mode: str) -> Optional[str]:
        mode_clean = str(mode).strip().lower()
        if mode_clean == "ti2v":
            return self.prompts.ti2v or self._legacy_prompt("ti2v_prompt", "prompt") or self.prompt_train or None
        if mode_clean == "ti2i":
            if self._legacy_prompt_disabled("ti2i_prompt"):
                return None
            return (
                self.prompts.ti2i
                or self._legacy_prompt("ti2i_prompt")
                or self.prompts.ti2v
                or self._legacy_prompt("ti2v_prompt", "prompt")
                or self.prompt_train
                or None
            )
        if mode_clean == "ti2t":
            if self._legacy_prompt_disabled("ti2t_prompt"):
                return None
            return self.prompts.ti2t or self._legacy_prompt("ti2t_prompt", "vlm_prompt", "gpt5_prompt") or None
        if mode_clean == "ti2ti":
            if self._legacy_prompt_disabled("ti2ti_prompt"):
                return None
            return (
                self.prompts.ti2ti
                or self._legacy_prompt("ti2ti_prompt")
                or self.prompts.ti2t
                or self._legacy_prompt("ti2t_prompt", "vlm_prompt", "gpt5_prompt")
                or None
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "task_group": self.task_group,
            "task_type": self.task_type,
            "prompts": self.prompts.to_dict(),
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
            prompts=CanonicalPrompts.from_dict(dict(payload.get("prompts") or {})),
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
