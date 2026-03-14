from __future__ import annotations

import json
import re


def _extract_answer_content(completion: str) -> str:
    start_tag = "<answer>"
    end_tag = "</answer>"
    text = completion or ""
    if start_tag in text and end_tag in text:
        try:
            start_idx = text.rfind(start_tag) + len(start_tag)
            end_idx = text.find(end_tag, start_idx)
            if end_idx != -1:
                text = text[start_idx:end_idx]
        except Exception:
            pass
    return text.strip()


def _normalize_free_text(text: str) -> str:
    normalized = _extract_answer_content(text).strip().lower()
    normalized = re.sub(r"^answer\s*:\s*", "", normalized)
    normalized = normalized.strip(" \t\r\n\"'`.,;:!?()[]{}")
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def reward_eyeballing(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        text = _normalize_free_text(completion)
        sol = _normalize_free_text(sol)
        if len(text) != 1 or not text.isalpha():
            rewards.append(-1.0)
            continue
        rewards.append(1.0 if text.upper() == sol.upper() else 0.0)
    return rewards


def reward_maze(completions, solution, **kwargs):
    rewards = []
    for completion, sol_str in zip(completions, solution):
        try:
            sol_path = json.loads(sol_str)
            total_len = len(sol_path)
            if total_len == 0:
                rewards.append(0.0)
                continue

            extract_content = _extract_answer_content(completion)
            match = re.search(r"\[(.*?)\]", extract_content, re.DOTALL)
            if not match:
                rewards.append(-1.0)
                continue

            content = match.group(1)
            pred_path = [
                int(x.strip())
                for x in content.split(",")
                if x.strip() and (x.strip().isdigit() or x.strip().lstrip("-").isdigit())
            ]
            if pred_path == sol_path:
                rewards.append(1.0)
                continue

            prefix_len = 0
            for p, s in zip(pred_path, sol_path):
                if p == s:
                    prefix_len += 1
                else:
                    break

            suffix_len = 0
            for p, s in zip(reversed(pred_path), reversed(sol_path)):
                if p == s:
                    suffix_len += 1
                else:
                    break

            effective_match = min(total_len, prefix_len + suffix_len)
            denom = max(total_len, len(pred_path))
            rewards.append(effective_match / denom)
        except Exception:
            rewards.append(-1.0)
    return rewards


def reward_visual_puzzle(completions, solution, **kwargs):
    rewards = []
    for completion, sol in zip(completions, solution):
        pred = _normalize_free_text(completion)
        target = _normalize_free_text(sol)
        if not pred:
            rewards.append(-1.0)
            continue
        rewards.append(1.0 if pred == target else 0.0)
    return rewards


def reward_format(completions, **kwargs):
    rewards = []
    for completion in completions:
        rewards.append(0.1 if completion.strip() else 0.0)
    return rewards
