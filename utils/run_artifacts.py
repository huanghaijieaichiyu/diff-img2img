from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunArtifactSummary:
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
    resolved_config: dict[str, Any] | None


def safe_float(value: str | None) -> float | None:
    if value in (None, "", "None"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: str | None) -> int | None:
    if value in (None, "", "None"):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def load_metrics_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def parse_metrics_txt(metrics_path: Path) -> tuple[dict[int, dict[str, float]], int | None]:
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
                train_step = safe_int(line.split(":", 1)[1].strip())
                continue
            if current_step_count is None or ":" not in line:
                continue
            key, raw_value = line.split(":", 1)
            value = safe_float(raw_value.strip())
            if value is None:
                continue
            normalized = key.strip().lower().replace(" ", "_")
            by_steps[current_step_count][normalized] = value
    return by_steps, train_step


def parse_status_json(status_path: Path) -> dict[str, float | int]:
    if not status_path.exists():
        return {}
    with status_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    parsed: dict[str, float | int] = {}
    for key in ("val_psnr", "val_ssim", "val_lpips", "step"):
        if key not in payload:
            continue
        value = safe_int(str(payload[key])) if key == "step" else safe_float(str(payload[key]))
        if value is not None:
            parsed[key] = value
    return parsed


def read_resolved_config(run_dir: Path) -> dict[str, Any] | None:
    config_path = run_dir / "resolved_config.json"
    if not config_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def choose_primary_metrics(
    run_dir: Path,
    *,
    preferred_step: int,
) -> tuple[Path | None, int | None, float | None, float | None, float | None, float | None]:
    candidates = [run_dir / "full_eval" / "metrics.txt", run_dir / "metrics.txt"]
    for metrics_path in candidates:
        metrics_by_steps, _ = parse_metrics_txt(metrics_path)
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

    status = parse_status_json(run_dir / "training_status.json")
    if status:
        return (
            run_dir / "training_status.json",
            preferred_step,
            safe_float(str(status.get("val_psnr"))),
            safe_float(str(status.get("val_ssim"))),
            safe_float(str(status.get("val_lpips"))),
            None,
        )
    return None, None, None, None, None, None


def tail_phase_rows(rows: list[dict[str, str]], *, window: int) -> tuple[list[dict[str, str]], str | None]:
    phase_priority = ("joint", "decom_warmup")
    for phase in phase_priority:
        phase_rows = [row for row in rows if row.get("phase") == phase]
        if phase_rows:
            return phase_rows[-window:] if len(phase_rows) > window else phase_rows, phase
    return (rows[-window:] if len(rows) > window else rows), None


def summarize_run_dir(run_dir: Path, *, window: int = 20, preferred_step: int = 8) -> RunArtifactSummary:
    csv_path = run_dir / "training_metrics.csv"
    rows = load_metrics_rows(csv_path)
    tail_rows, sampled_phase = tail_phase_rows(rows, window=window)
    throughput = [safe_float(row.get("samples_per_sec")) for row in tail_rows]
    wait_ratio = [safe_float(row.get("data_wait_ratio")) for row in tail_rows]
    cpu_percent = [safe_float(row.get("cpu_percent")) for row in tail_rows]
    gpu_reserved = [safe_float(row.get("gpu_reserved_gb")) for row in tail_rows]

    def clean(values: list[float | None]) -> list[float]:
        return [value for value in values if value is not None]

    metrics_source, primary_step, psnr, ssim, lpips, seconds_per_image = choose_primary_metrics(
        run_dir,
        preferred_step=preferred_step,
    )
    latest_step = safe_int(rows[-1].get("step")) if rows else None
    gpu_reserved_values = clean(gpu_reserved)
    return RunArtifactSummary(
        name=run_dir.name,
        run_dir=run_dir,
        metrics_source=metrics_source,
        primary_step=primary_step,
        psnr=psnr,
        ssim=ssim,
        lpips=lpips,
        seconds_per_image=seconds_per_image,
        throughput_mean=mean(clean(throughput)),
        wait_ratio_mean=mean(clean(wait_ratio)),
        cpu_percent_mean=mean(clean(cpu_percent)),
        gpu_reserved_mean=mean(gpu_reserved_values),
        gpu_reserved_max=max(gpu_reserved_values) if gpu_reserved_values else None,
        sampled_rows=len(tail_rows),
        sampled_phase=sampled_phase,
        latest_step=latest_step,
        resolved_config=read_resolved_config(run_dir),
    )


def delta_ratio(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None or baseline == 0:
        return None
    return (current / baseline) - 1.0


def delta(current: float | None, baseline: float | None) -> float | None:
    if current is None or baseline is None:
        return None
    return current - baseline


def fmt_value(value: float | int | None, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if math.isnan(value):
        return "-"
    return f"{value:.{digits}f}"


def summary_lines(summary: RunArtifactSummary) -> list[str]:
    lines = [
        f"[{summary.name}]",
        f"run_dir: {summary.run_dir}",
        f"metrics_source: {summary.metrics_source if summary.metrics_source is not None else '-'}",
        f"sampled_phase: {summary.sampled_phase or '-'}",
        f"sampled_rows: {summary.sampled_rows}",
        f"latest_step: {summary.latest_step if summary.latest_step is not None else '-'}",
        f"primary_step: {summary.primary_step if summary.primary_step is not None else '-'}",
        f"samples_per_sec_mean: {fmt_value(summary.throughput_mean, 3)}",
        f"data_wait_ratio_mean: {fmt_value(summary.wait_ratio_mean, 3)}",
        f"cpu_percent_mean: {fmt_value(summary.cpu_percent_mean, 1)}",
        f"gpu_reserved_mean_gb: {fmt_value(summary.gpu_reserved_mean, 2)}",
        f"gpu_reserved_max_gb: {fmt_value(summary.gpu_reserved_max, 2)}",
        f"psnr: {fmt_value(summary.psnr, 4)}",
        f"ssim: {fmt_value(summary.ssim, 4)}",
        f"lpips: {fmt_value(summary.lpips, 4)}",
        f"seconds_per_image: {fmt_value(summary.seconds_per_image, 6)}",
    ]
    if summary.resolved_config and summary.resolved_config.get("summary"):
        config_summary = summary.resolved_config["summary"]
        lines.append(
            "config_summary: "
            f"config={config_summary.get('config_name')} "
            f"effective_batch={config_summary.get('effective_batch_size')} "
            f"resolution={config_summary.get('resolution')} "
            f"decom={config_summary.get('decom_variant')} "
            f"condition={config_summary.get('condition_variant')}"
        )
    return lines
