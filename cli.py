from __future__ import annotations

import argparse

from data import export as data_export
from data import generate as data_generate
from evaluation import commands as eval_commands


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="VideoThinkBench unified data and evaluation toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data_parser = subparsers.add_parser("data", help="Data generation and export")
    data_subparsers = data_parser.add_subparsers(dest="data_command", required=True)
    data_generate.build_parser(data_subparsers)
    data_export.build_parser(data_subparsers)

    eval_commands.build_parser(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return
    func(args)


if __name__ == "__main__":
    main()
