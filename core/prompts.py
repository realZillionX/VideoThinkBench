from __future__ import annotations

import re
from typing import Optional

from data.registry import VISUAL_PUZZLE_TASKS


MAZE_TRAIN_PROMPT = "Use the provided maze image as input. Draw a red path connecting the two red endpoints without touching the black walls."

VLM_SFT_SUFFIX = "\nDo not output the thinking process. Output the answer directly."
VLM_GRPO_SUFFIX = "\nPlease think step by step and output your final answer within <answer>...</answer> tags."


def normalize_eyeballing_prompt(prompt: str) -> str:
    patterns_to_remove = [
        r"\s*Speak out[^.]*\.[^.]*\.",
        r"\s*In portrait[^.]*\.",
        r"\s*Static camera\.",
    ]
    result = prompt
    for pattern in patterns_to_remove:
        result = re.sub(pattern, "", result, flags=re.IGNORECASE)
    result = " ".join(result.split())
    if result and not result.endswith("."):
        result += "."
    return result.strip()


def _normalize_prompt_text(prompt: Optional[str]) -> str:
    text = " ".join(str(prompt or "").split())
    return text.strip()


def ensure_image_conditioned_prompt(prompt: Optional[str], *, mode: str) -> str:
    normalized = _normalize_prompt_text(prompt)
    if not normalized:
        return ""
    lowered = normalized.lower()
    if "provided puzzle image" in lowered or "provided maze image" in lowered or "provided input image" in lowered:
        return normalized
    if mode == "ti2v":
        return f"Use the provided puzzle image as the starting frame. {normalized}".strip()
    if mode == "ti2i":
        return f"Use the provided puzzle image as input. {normalized}".strip()
    if mode == "ti2ti":
        return f"Use the provided reasoning image as input. {normalized}".strip()
    if mode == "ti2t":
        return f"Use the provided puzzle image to solve the task. {normalized}".strip()
    return normalized


def normalize_prompt_for_task(task_group: str, prompt_raw: str) -> str:
    if task_group == "maze":
        return MAZE_TRAIN_PROMPT
    if task_group == "eyeballing":
        return ensure_image_conditioned_prompt(normalize_eyeballing_prompt(prompt_raw), mode="ti2i")
    if task_group == "visual_puzzle":
        return ensure_image_conditioned_prompt(prompt_raw, mode="ti2v")
    return ensure_image_conditioned_prompt(prompt_raw, mode="ti2i")


def build_vlm_user_prompt(prompt_train: str, mode: str) -> str:
    suffix = VLM_SFT_SUFFIX if mode == "sft" else VLM_GRPO_SUFFIX
    return f"<image> {prompt_train}{suffix}".strip()


def detect_task_group(record: dict, puzzle_name: Optional[str] = None) -> Optional[str]:
    task_type = str(record.get("task_type") or puzzle_name or "")
    if task_type in VISUAL_PUZZLE_TASKS:
        return "visual_puzzle"
    if "correct_option" in record:
        return "eyeballing"
    if "solution_path_cell_ids" in record:
        return "maze"
    if task_type.startswith("maze"):
        return "maze"
    return None
