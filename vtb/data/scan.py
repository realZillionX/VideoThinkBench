from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from vtb.schemas import CanonicalAnswer, CanonicalAssets, CanonicalSample
from vtb.utils.io import read_json, read_jsonl, write_jsonl
from vtb.utils.paths import to_absolute
from vtb.utils.prompts import detect_task_group, normalize_prompt_for_task


def build_canonical_sample(
    record: Dict,
    *,
    task_dir: Path,
    metadata_path: Path,
    source_task_name: str,
) -> Optional[CanonicalSample]:
    task_group = detect_task_group(record, puzzle_name=source_task_name)
    if task_group is None:
        return None

    task_type = str(record.get("task_type") or source_task_name)
    sample_id = str(record.get("id") or f"{task_type}:{record.get('image', '')}")
    prompt_raw = str(record.get("prompt") or record.get("gpt5_prompt") or "").strip()
    prompt_train = normalize_prompt_for_task(task_group, prompt_raw)

    image_value = record.get("image")
    solution_image_value = record.get("solution_image_path")
    if not image_value or not solution_image_value:
        return None

    puzzle_image = to_absolute(str(image_value), task_dir).as_posix()
    solution_image = to_absolute(str(solution_image_value), task_dir).as_posix()

    solution_video = None
    raw_video = record.get("solution_video_path") or record.get("video")
    if raw_video:
        solution_video = to_absolute(str(raw_video), task_dir).as_posix()

    if task_group == "maze":
        path_ids = record.get("solution_path_cell_ids")
        if not isinstance(path_ids, list):
            return None
        answer = CanonicalAnswer(path_cell_ids=[int(item) for item in path_ids])
    else:
        correct_option = str(record.get("correct_option") or "").strip().upper()
        if len(correct_option) != 1:
            return None
        answer = CanonicalAnswer(correct_option=correct_option)

    return CanonicalSample(
        id=sample_id,
        task_group=task_group,
        task_type=task_type,
        prompt_raw=prompt_raw,
        prompt_train=prompt_train,
        assets=CanonicalAssets(
            puzzle_image=puzzle_image,
            solution_image=solution_image,
            solution_video=solution_video,
        ),
        answer=answer,
        source={
            "task": source_task_name,
            "metadata_path": metadata_path.as_posix(),
            "task_dir": task_dir.as_posix(),
        },
        extra={
            "raw_record": record,
        },
    )


def dedup_samples(samples: Sequence[CanonicalSample]) -> Tuple[List[CanonicalSample], int]:
    deduped: Dict[Tuple[str, str, str], CanonicalSample] = {}
    dropped = 0
    for sample in samples:
        key = (sample.id, sample.task_group, sample.assets.puzzle_image)
        if key in deduped:
            dropped += 1
            continue
        deduped[key] = sample
    return list(deduped.values()), dropped


def filter_by_task_group(samples: Iterable[CanonicalSample], task_groups: Sequence[str]) -> List[CanonicalSample]:
    allowed = set(task_groups)
    filtered: List[CanonicalSample] = []
    for sample in samples:
        if sample.task_group is None:
            continue
        if sample.task_group not in allowed:
            continue
        filtered.append(sample)
    return filtered


def write_manifest(path: Path, samples: Sequence[CanonicalSample]) -> None:
    write_jsonl(path, [sample.to_dict() for sample in samples])


def load_manifest(path: Path) -> List[CanonicalSample]:
    rows = read_jsonl(path)
    return [CanonicalSample.from_dict(row) for row in rows]


def load_task_data_json(metadata_path: Path) -> List[Dict]:
    payload = read_json(metadata_path)
    if not isinstance(payload, list):
        raise ValueError(f"Metadata is not a list: {metadata_path}")
    return payload


def iter_metadata_files(data_root: Path) -> Iterable[Path]:
    for metadata_path in sorted(data_root.rglob("data.json")):
        yield metadata_path


def build_samples_from_data_root(
    data_root: Path,
    *,
    task_groups: Sequence[str],
) -> List[CanonicalSample]:
    samples: List[CanonicalSample] = []
    allowed = set(task_groups)
    for metadata_path in iter_metadata_files(data_root):
        task_dir = metadata_path.parent
        source_task_name = task_dir.name
        payload = load_task_data_json(metadata_path)
        for record in payload:
            if not isinstance(record, dict):
                continue
            sample = build_canonical_sample(
                record,
                task_dir=task_dir,
                metadata_path=metadata_path,
                source_task_name=source_task_name,
            )
            if sample is None:
                continue
            if sample.task_group not in allowed:
                continue
            samples.append(sample)
    deduped, _ = dedup_samples(samples)
    return deduped
