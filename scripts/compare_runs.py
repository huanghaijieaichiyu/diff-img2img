#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.run_artifacts import RunArtifactSummary, fmt_value, summarize_run_dir


SORT_FIELDS = {
    "psnr": True,
    "ssim": True,
    "lpips": False,
    "throughput": True,
    "wait_ratio": False,
    "seconds_per_image": False,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare multiple Diff-Img2Img run directories.")
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--preferred-step", type=int, default=8)
    parser.add_argument("--sort-by", choices=sorted(SORT_FIELDS), default="psnr")
    return parser


def sort_value(summary: RunArtifactSummary, sort_by: str) -> float:
    mapping = {
        "psnr": summary.psnr,
        "ssim": summary.ssim,
        "lpips": summary.lpips,
        "throughput": summary.throughput_mean,
        "wait_ratio": summary.wait_ratio_mean,
        "seconds_per_image": summary.seconds_per_image,
    }
    value = mapping.get(sort_by)
    if value is None:
        return float("-inf") if SORT_FIELDS[sort_by] else float("inf")
    return value


def main() -> int:
    args = build_parser().parse_args()
    summaries = [
        summarize_run_dir(run_dir.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
        for run_dir in args.run_dir
    ]
    reverse = SORT_FIELDS[args.sort_by]
    summaries.sort(key=lambda summary: sort_value(summary, args.sort_by), reverse=reverse)

    print(
        "run\tphase\tstep\tpsnr\tssim\tlpips\tsamples_per_sec\twait_ratio\tseconds_per_image\tconfig\teffective_batch"
    )
    for summary in summaries:
        config = (summary.resolved_config or {}).get("summary", {})
        print(
            "\t".join(
                [
                    summary.name,
                    summary.sampled_phase or "-",
                    str(summary.latest_step or "-"),
                    fmt_value(summary.psnr, 4),
                    fmt_value(summary.ssim, 4),
                    fmt_value(summary.lpips, 4),
                    fmt_value(summary.throughput_mean, 3),
                    fmt_value(summary.wait_ratio_mean, 3),
                    fmt_value(summary.seconds_per_image, 6),
                    str(config.get("config_name", "-")),
                    str(config.get("effective_batch_size", "-")),
                ]
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
