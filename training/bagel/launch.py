from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch Bagel training with VideoThinkBench dataset metadata.",
        add_help=True,
    )
    parser.add_argument("--bagel-path", required=True, help="Path to the cloned Bagel repository.")
    parser.add_argument("--dataset-info-json", required=True, help="Path to exported Bagel dataset_info.json.")
    return parser


def main() -> None:
    parser = build_parser()
    args, remaining = parser.parse_known_args()

    bagel_path = Path(args.bagel_path).expanduser().resolve()
    if not bagel_path.exists():
        raise FileNotFoundError(f"Bagel path not found: {bagel_path}")
    dataset_info_json = Path(args.dataset_info_json).expanduser().resolve()
    if not dataset_info_json.exists():
        raise FileNotFoundError(f"Bagel dataset info not found: {dataset_info_json}")

    if bagel_path.as_posix() not in sys.path:
        sys.path.insert(0, bagel_path.as_posix())

    dataset_info_module = importlib.import_module("data.dataset_info")
    with dataset_info_json.open("r", encoding="utf-8") as handle:
        dataset_info_payload = json.load(handle)
    dataset_info_module.DATASET_INFO = dataset_info_payload

    train_module = importlib.import_module("train.pretrain_unified_navit")
    sys.argv = [str(bagel_path / "train" / "pretrain_unified_navit.py"), *remaining]
    train_module.main()


if __name__ == "__main__":
    main()
