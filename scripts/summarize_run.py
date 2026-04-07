#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.run_artifacts import summary_lines, summarize_run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize a Diff-Img2Img run directory.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--preferred-step", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    summary = summarize_run_dir(args.run_dir.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
    for line in summary_lines(summary):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
