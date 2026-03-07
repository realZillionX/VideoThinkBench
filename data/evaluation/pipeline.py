from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable, List

from core.schemas import EvalRecord
from core.io import write_json, write_jsonl


def write_eval_outputs(output_dir: Path, records: Iterable[EvalRecord]) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [record.to_dict() for record in records]
    results_path = output_dir / "results.jsonl"
    write_jsonl(results_path, rows)

    total = len(rows)
    errors = sum(1 for row in rows if row.get("error"))
    pass_count = sum(1 for row in rows if row.get("offline_pass") is True)
    fail_count = sum(1 for row in rows if row.get("offline_pass") is False)
    by_group = Counter(row.get("task_group", "unknown") for row in rows)
    by_type = Counter(row.get("task_type", "unknown") for row in rows)

    summary = {
        "total": total,
        "pass": pass_count,
        "fail": fail_count,
        "error": errors,
        "by_group": dict(by_group),
        "by_type": dict(by_type),
        "results_path": results_path.as_posix(),
    }
    summary_path = output_dir / "summary.json"
    write_json(summary_path, summary)
    summary["summary_path"] = summary_path.as_posix()
    return summary
