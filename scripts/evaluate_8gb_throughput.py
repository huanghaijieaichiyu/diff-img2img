#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.run_artifacts import (
    RunArtifactSummary,
    delta,
    delta_ratio,
    fmt_value as _fmt,
    summary_lines,
    summarize_run_dir,
)


def _phase_a_passes(baseline: RunArtifactSummary, phase_a: RunArtifactSummary) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    throughput_gain = delta_ratio(phase_a.throughput_mean, baseline.throughput_mean)
    if throughput_gain is None or throughput_gain < 0.20:
        reasons.append("throughput gain < 20%")
    if phase_a.wait_ratio_mean is None or phase_a.wait_ratio_mean > 0.10:
        reasons.append("data_wait_ratio > 0.10")
    psnr_delta = delta(phase_a.psnr, baseline.psnr)
    if psnr_delta is None or psnr_delta < -0.10:
        reasons.append("PSNR drop > 0.10 dB")
    ssim_delta = delta(phase_a.ssim, baseline.ssim)
    if ssim_delta is None or ssim_delta < -0.003:
        reasons.append("SSIM drop > 0.003")
    lpips_ratio = delta_ratio(phase_a.lpips, baseline.lpips)
    if lpips_ratio is None or lpips_ratio > 0.02:
        reasons.append("LPIPS worsened by > 2%")
    return not reasons, reasons


def _phase_b_passes(phase_a: RunArtifactSummary, phase_b: RunArtifactSummary) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    throughput_gain = delta_ratio(phase_b.throughput_mean, phase_a.throughput_mean)
    if throughput_gain is None or throughput_gain < 0.35:
        reasons.append("throughput gain < 35%")
    psnr_delta = delta(phase_b.psnr, phase_a.psnr)
    if psnr_delta is None or psnr_delta < -0.20:
        reasons.append("PSNR drop > 0.20 dB")
    ssim_delta = delta(phase_b.ssim, phase_a.ssim)
    if ssim_delta is None or ssim_delta < -0.005:
        reasons.append("SSIM drop > 0.005")
    lpips_ratio = delta_ratio(phase_b.lpips, phase_a.lpips)
    if lpips_ratio is None or lpips_ratio > 0.03:
        reasons.append("LPIPS worsened by > 3%")
    return not reasons, reasons


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate baseline / Phase A / Phase B 8GB throughput runs against the plan thresholds."
    )
    parser.add_argument("--baseline-run", type=Path, required=True)
    parser.add_argument("--phase-a-run", type=Path, required=True)
    parser.add_argument("--phase-b-run", type=Path, action="append", default=[])
    parser.add_argument("--window", type=int, default=20, help="How many trailing training rows to average.")
    parser.add_argument("--preferred-step", type=int, default=8, help="Preferred validation step count inside metrics.txt.")
    parser.add_argument("--cpu-high-threshold", type=float, default=85.0)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    baseline = summarize_run_dir(args.baseline_run.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
    phase_a = summarize_run_dir(args.phase_a_run.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
    phase_b_runs = [
        summarize_run_dir(path.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
        for path in args.phase_b_run
    ]

    for line in summary_lines(baseline):
        print(line)
    print()
    for line in summary_lines(phase_a):
        print(line)
    print()
    for summary in phase_b_runs:
        for line in summary_lines(summary):
            print(line)
        print()

    phase_a_ok, phase_a_reasons = _phase_a_passes(baseline, phase_a)
    throughput_gain_a = delta_ratio(phase_a.throughput_mean, baseline.throughput_mean)
    print("Phase A verdict:")
    print(f"pass: {'yes' if phase_a_ok else 'no'}")
    print(f"throughput_gain: {_fmt(throughput_gain_a * 100 if throughput_gain_a is not None else None, 2)}%")
    if phase_a_reasons:
        print("reasons: " + "; ".join(phase_a_reasons))
    print()

    best_phase_b: RunArtifactSummary | None = None
    best_phase_b_gain: float | None = None
    for summary in phase_b_runs:
        phase_b_ok, phase_b_reasons = _phase_b_passes(phase_a, summary)
        throughput_gain_b = delta_ratio(summary.throughput_mean, phase_a.throughput_mean)
        print(f"Phase B verdict for {summary.name}:")
        print(f"pass: {'yes' if phase_b_ok else 'no'}")
        print(f"throughput_gain_vs_phase_a: {_fmt(throughput_gain_b * 100 if throughput_gain_b is not None else None, 2)}%")
        if phase_b_reasons:
            print("reasons: " + "; ".join(phase_b_reasons))
        print()
        if phase_b_ok:
            if best_phase_b is None or (throughput_gain_b is not None and (best_phase_b_gain is None or throughput_gain_b > best_phase_b_gain)):
                best_phase_b = summary
                best_phase_b_gain = throughput_gain_b

    print("Recommendation:")
    if not phase_a_ok:
        if phase_a.wait_ratio_mean is not None and phase_a.wait_ratio_mean > 0.15:
            if phase_a.cpu_percent_mean is None or phase_a.cpu_percent_mean < args.cpu_high_threshold:
                print("Phase A did not pass. Try Phase C with `middle_phase_c_workers8` next.")
            else:
                print("Phase A did not pass. CPU already looks saturated, so try Phase C with `middle_phase_c_prefetch2_cache32` next.")
        else:
            print("Phase A did not pass, but the bottleneck does not look loader-bound. Keep `middle_phase_a` as the base and inspect model/validation cost before Phase B.")
        return 0

    if not phase_b_runs:
        print("Phase A passed. Run the Phase B candidates next and compare them against Phase A.")
        return 0

    if best_phase_b is not None:
        print(f"Use `{best_phase_b.name}` as the preferred 8GB candidate. It passed the Phase B thresholds and delivered the strongest throughput gain.")
    else:
        print("None of the Phase B candidates passed. Stay on `middle_phase_a` as the default recommendation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
