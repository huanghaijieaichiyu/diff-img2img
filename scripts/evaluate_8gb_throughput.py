#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunSummary:
    name: str
    run_dir: Path
    metrics_source: Path | None
    primary_step: int | None
    psnr: float | None
    ssim: float | None
    lpips: float | None
    seconds_per_image: float | None
    throughput_mean: float | None
    wait_ratio_mean: float | None
    cpu_percent_mean: float | None
    gpu_reserved_mean: float | None
    gpu_reserved_max: float | None
    sampled_rows: int
    sampled_phase: str | None
    latest_step: int | None


def _safe_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _load_metrics_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def _parse_metrics_txt(metrics_path: Path) -> tuple[dict[int, dict[str, float]], int | None]:
    if not metrics_path.exists():
        return {}, None

    by_steps: dict[int, dict[str, float]] = {}
    current_step_count: int | None = None
    train_step: int | None = None

    with metrics_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("[steps=") and line.endswith("]"):
                current_step_count = int(line[len("[steps="):-1])
                by_steps[current_step_count] = {}
                continue
            if line.startswith("Step:"):
                train_step = _safe_int(line.split(":", 1)[1].strip())
                continue
            if current_step_count is None or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            value = _safe_float(raw_value.strip())
            if value is None:
                continue
            normalized = key.strip().lower().replace(" ", "_")
            by_steps[current_step_count][normalized] = value
    return by_steps, train_step


def _parse_status_json(status_path: Path) -> dict[str, float | int]:
    if not status_path.exists():
        return {}
    with status_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    parsed: dict[str, float | int] = {}
    for key in ("val_psnr", "val_ssim", "val_lpips", "step"):
        if key not in payload:
            continue
        if key == "step":
            value = _safe_int(str(payload[key]))
        else:
            value = _safe_float(str(payload[key]))
        if value is not None:
            parsed[key] = value
    return parsed


def _choose_primary_metrics(
    run_dir: Path,
    *,
    preferred_step: int,
) -> tuple[Path | None, int | None, float | None, float | None, float | None, float | None]:
    candidates = [run_dir / "full_eval" / "metrics.txt", run_dir / "metrics.txt"]
    for metrics_path in candidates:
        metrics_by_steps, _ = _parse_metrics_txt(metrics_path)
        if not metrics_by_steps:
            continue
        selected_step = preferred_step if preferred_step in metrics_by_steps else sorted(metrics_by_steps)[0]
        metrics = metrics_by_steps[selected_step]
        return (
            metrics_path,
            selected_step,
            metrics.get("psnr"),
            metrics.get("ssim"),
            metrics.get("lpips"),
            metrics.get("secondsperimage"),
        )

    status = _parse_status_json(run_dir / "training_status.json")
    if status:
        return (
            run_dir / "training_status.json",
            preferred_step,
            _safe_float(str(status.get("val_psnr"))),
            _safe_float(str(status.get("val_ssim"))),
            _safe_float(str(status.get("val_lpips"))),
            None,
        )
    return None, None, None, None, None, None


def _tail_phase_rows(rows: list[dict[str, str]], *, window: int) -> tuple[list[dict[str, str]], str | None]:
    phase_priority = ("joint", "decom_warmup")
    for phase in phase_priority:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        if phase_rows:
            return phase_rows[-window:] if len(phase_rows) > window else phase_rows, phase
    return (rows[-window:] if len(rows) > window else rows), None


def summarize_run(run_dir: Path, *, window: int, preferred_step: int) -> RunSummary:
    csv_path = run_dir / "training_metrics.csv"
    rows = _load_metrics_rows(csv_path)
    tail_rows, sampled_phase = _tail_phase_rows(rows, window=window)
    throughput = [_safe_float(row.get("samples_per_sec")) for row in tail_rows]
    wait_ratio = [_safe_float(row.get("data_wait_ratio")) for row in tail_rows]
    cpu_percent = [_safe_float(row.get("cpu_percent")) for row in tail_rows]
    gpu_reserved = [_safe_float(row.get("gpu_reserved_gb")) for row in tail_rows]

    def clean(values: list[float | None]) -> list[float]:
        return [value for value in values if value is not None]

    metrics_source, primary_step, psnr, ssim, lpips, seconds_per_image = _choose_primary_metrics(
        run_dir,
        preferred_step=preferred_step,
    )
    latest_step = None
    if rows:
        latest_step = _safe_int(rows[-1].get("step"))

    gpu_reserved_values = clean(gpu_reserved)
    return RunSummary(
        name=run_dir.name,
        run_dir=run_dir,
        metrics_source=metrics_source,
        primary_step=primary_step,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        seconds_per_image=seconds_per_image,
        throughput_mean=_mean(clean(throughput)),
        wait_ratio_mean=_mean(clean(wait_ratio)),
        cpu_percent_mean=_mean(clean(cpu_percent)),
        gpu_reserved_mean=_mean(gpu_reserved_values),
        gpu_reserved_max=max(gpu_reserved_values) if gpu_reserved_values else None,
        sampled_rows=len(tail_rows),
        sampled_phase=sampled_phase,
        latest_step=latest_step,
    )


def _delta_ratio(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current / baseline) - 1.0


def _delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def _phase_a_passes(baseline: RunSummary, phase_a: RunSummary) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    throughput_gain = _delta_ratio(phase_a.throughput_mean, baseline.throughput_mean)
    if throughput_gain is None or throughput_gain < 0.20:
        reasons.append("throughput gain < 20%")
    if phase_a.wait_ratio_mean is None or phase_a.wait_ratio_mean > 0.10:
        reasons.append("data_wait_ratio > 0.10")
    psnr_delta = _delta(phase_a.psnr, baseline.psnr)
    if psnr_delta is None or psnr_delta < -0.10:
        reasons.append("PSNR drop > 0.10 dB")
    ssim_delta = _delta(phase_a.ssim, baseline.ssim)
    if ssim_delta is None or ssim_delta < -0.003:
        reasons.append("SSIM drop > 0.003")
    lpips_ratio = _delta_ratio(phase_a.lpips, baseline.lpips)
    if lpips_ratio is None or lpips_ratio > 0.02:
        reasons.append("LPIPS worsened by > 2%")
    return not reasons, reasons


def _phase_b_passes(phase_a: RunSummary, phase_b: RunSummary) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    throughput_gain = _delta_ratio(phase_b.throughput_mean, phase_a.throughput_mean)
    if throughput_gain is None or throughput_gain < 0.35:
        reasons.append("throughput gain < 35%")
    psnr_delta = _delta(phase_b.psnr, phase_a.psnr)
    if psnr_delta is None or psnr_delta < -0.20:
        reasons.append("PSNR drop > 0.20 dB")
    ssim_delta = _delta(phase_b.ssim, phase_a.ssim)
    if ssim_delta is None or ssim_delta < -0.005:
        reasons.append("SSIM drop > 0.005")
    lpips_ratio = _delta_ratio(phase_b.lpips, phase_a.lpips)
    if lpips_ratio is None or lpips_ratio > 0.03:
        reasons.append("LPIPS worsened by > 3%")
    return not reasons, reasons


def _fmt(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def _print_summary(summary: RunSummary) -> None:
    print(f"[{summary.name}]")
    print(f"run_dir: {summary.run_dir}")
    print(f"metrics_source: {summary.metrics_source if summary.metrics_source is not None else '-'}")
    print(f"sampled_phase: {summary.sampled_phase or '-'}")
    print(f"sampled_rows: {summary.sampled_rows}")
    print(f"latest_step: {summary.latest_step if summary.latest_step is not None else '-'}")
    print(f"primary_step: {summary.primary_step if summary.primary_step is not None else '-'}")
    print(f"samples_per_sec_mean: {_fmt(summary.throughput_mean, 3)}")
    print(f"data_wait_ratio_mean: {_fmt(summary.wait_ratio_mean, 3)}")
    print(f"cpu_percent_mean: {_fmt(summary.cpu_percent_mean, 1)}")
    print(f"gpu_reserved_mean_gb: {_fmt(summary.gpu_reserved_mean, 2)}")
    print(f"gpu_reserved_max_gb: {_fmt(summary.gpu_reserved_max, 2)}")
    print(f"psnr: {_fmt(summary.psnr, 4)}")
    print(f"ssim: {_fmt(summary.ssim, 4)}")
    print(f"lpips: {_fmt(summary.lpips, 4)}")
    print(f"seconds_per_image: {_fmt(summary.seconds_per_image, 6)}")
    print()


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

    baseline = summarize_run(args.baseline_run.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
    phase_a = summarize_run(args.phase_a_run.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
    phase_b_runs = [
        summarize_run(path.resolve(), window=max(1, args.window), preferred_step=args.preferred_step)
        for path in args.phase_b_run
    ]

    _print_summary(baseline)
    _print_summary(phase_a)
    for summary in phase_b_runs:
        _print_summary(summary)

    phase_a_ok, phase_a_reasons = _phase_a_passes(baseline, phase_a)
    throughput_gain_a = _delta_ratio(phase_a.throughput_mean, baseline.throughput_mean)
    print("Phase A verdict:")
    print(f"pass: {'yes' if phase_a_ok else 'no'}")
    print(f"throughput_gain: {_fmt(throughput_gain_a * 100 if throughput_gain_a is not None else None, 2)}%")
    if phase_a_reasons:
        print("reasons: " + "; ".join(phase_a_reasons))
    print()

    best_phase_b: RunSummary | None = None
    best_phase_b_gain: float | None = None
    passing_phase_b: list[tuple[RunSummary, float | None]] = []
    for summary in phase_b_runs:
        phase_b_ok, phase_b_reasons = _phase_b_passes(phase_a, summary)
        throughput_gain_b = _delta_ratio(summary.throughput_mean, phase_a.throughput_mean)
        print(f"Phase B verdict for {summary.name}:")
        print(f"pass: {'yes' if phase_b_ok else 'no'}")
        print(f"throughput_gain_vs_phase_a: {_fmt(throughput_gain_b * 100 if throughput_gain_b is not None else None, 2)}%")
        if phase_b_reasons:
            print("reasons: " + "; ".join(phase_b_reasons))
        print()
        if phase_b_ok:
            passing_phase_b.append((summary, throughput_gain_b))
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
