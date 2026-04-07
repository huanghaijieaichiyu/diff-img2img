#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"
DEFAULT_ACCELERATE_CONFIG = PROJECT_ROOT / "accelerate_config.yaml"
DEFAULT_PYTHON_CANDIDATES = (
    PROJECT_ROOT / ".venv" / "bin" / "python3",
    PROJECT_ROOT / ".venv" / "bin" / "python",
)


PHASE_A_OVERRIDES = (
    "--num_workers", "6",
    "--prefetch_factor", "4",
    "--persistent_workers",
    "--pin_memory",
    "--decode_cache_size", "48",
    "--opencv_threads_per_worker", "1",
    "--prepare_workers", "8",
    "--validation_steps", "1000",
    "--num_validation_images", "6",
    "--benchmark_inference_steps", "8",
    "--semantic_backbone", "none",
    "--nr_metric", "none",
)

PHASE_C_CPU_FREE_OVERRIDES = (
    "--num_workers", "8",
    "--prefetch_factor", "4",
    "--persistent_workers",
    "--pin_memory",
    "--decode_cache_size", "48",
    "--opencv_threads_per_worker", "1",
    "--prepare_workers", "8",
    "--validation_steps", "1000",
    "--num_validation_images", "6",
    "--benchmark_inference_steps", "8",
    "--semantic_backbone", "none",
    "--nr_metric", "none",
)

PHASE_C_CPU_BOUND_OVERRIDES = (
    "--num_workers", "6",
    "--prefetch_factor", "2",
    "--persistent_workers",
    "--pin_memory",
    "--decode_cache_size", "32",
    "--opencv_threads_per_worker", "1",
    "--prepare_workers", "8",
    "--validation_steps", "1000",
    "--num_validation_images", "6",
    "--benchmark_inference_steps", "8",
    "--semantic_backbone", "none",
    "--nr_metric", "none",
)

FULL_VALIDATE_OVERRIDES = (
    "--batch_size", "2",
    "--num_validation_images", "12",
    "--benchmark_inference_steps", "8", "20",
    "--semantic_backbone", "resnet18",
    "--nr_metric", "niqe",
)


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    config_name: str
    description: str
    max_train_steps: int
    overrides: tuple[str, ...]


def _default_python_executable() -> Path:
    for candidate in DEFAULT_PYTHON_CANDIDATES:
        if candidate.exists():
            return candidate
    return Path(sys.executable)


def _local_storage_warning(data_dir: Path, prepared_cache_dir: Path | None) -> str | None:
    candidates = [data_dir]
    if prepared_cache_dir is not None:
        candidates.append(prepared_cache_dir)
    else:
        candidates.append(data_dir / ".prepared")

    slow_paths = [str(path) for path in candidates if str(path).startswith("/mnt/")]
    if not slow_paths:
        return None
    joined = ", ".join(slow_paths)
    return (
        "warning: the plan assumes dataset + prepared cache live on a local Linux SSD. "
        f"These paths look cross-mounted and may distort throughput numbers: {joined}"
    )


def _quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _accelerate_prefix(accelerate_config: Path | None, python_executable: Path) -> list[str]:
    prefix = [str(python_executable), "-m", "accelerate.commands.launch"]
    if accelerate_config is not None:
        prefix.extend(["--config_file", str(accelerate_config)])
    return prefix


def _config_path(config_name: str) -> Path:
    return PROJECT_ROOT / "configs" / "train" / f"{config_name}.yaml"


def _train_command(
    spec: ExperimentSpec,
    *,
    data_dir: Path,
    output_root: Path,
    train_profile: str,
    accelerate_config: Path | None,
    python_executable: Path,
    prepared_cache_dir: Path | None,
) -> list[str]:
    output_dir = output_root / spec.name
    command = _accelerate_prefix(accelerate_config, python_executable)
    command.extend(
        [
            str(MAIN_PY),
            "--mode", "train",
            "--config", str(_config_path(spec.config_name)),
            "--data_dir", str(data_dir),
            "--output_dir", str(output_dir),
            "--train_profile", train_profile,
            "--use_retinex",
            "--max_train_steps", str(spec.max_train_steps),
        ]
    )
    if prepared_cache_dir is not None:
        command.extend(["--prepared_cache_dir", str(prepared_cache_dir)])
    command.extend(spec.overrides)
    return command


def _validate_command(
    spec: ExperimentSpec,
    *,
    data_dir: Path,
    output_root: Path,
    accelerate_config: Path | None,
    python_executable: Path,
) -> list[str]:
    run_dir = output_root / spec.name
    model_path = run_dir / "best_model"
    eval_dir = run_dir / "full_eval"
    command = _accelerate_prefix(accelerate_config, python_executable)
    command.extend(
        [
            str(MAIN_PY),
            "--mode", "validate",
            "--config", str(_config_path(spec.config_name)),
            "--data_dir", str(data_dir),
            "--model_path", str(model_path),
            "--output_dir", str(eval_dir),
        ]
    )
    command.extend(FULL_VALIDATE_OVERRIDES)
    return command


def _specs(benchmark_steps: int) -> dict[str, ExperimentSpec]:
    phase_b_steps = max(1, round(benchmark_steps * 1.5))
    return {
        "middle_baseline": ExperimentSpec(
            name="middle_baseline",
            config_name="middle",
            description="Original middle preset with only max_train_steps pinned for the benchmark window.",
            max_train_steps=benchmark_steps,
            overrides=(),
        ),
        "middle_phase_a": ExperimentSpec(
            name="middle_phase_a",
            config_name="middle",
            description="Low-risk 8GB throughput baseline: reduce validation cost and tune loader/cache settings only.",
            max_train_steps=benchmark_steps,
            overrides=PHASE_A_OVERRIDES,
        ),
        "small_phase_b_batch4": ExperimentSpec(
            name="small_phase_b_batch4",
            config_name="small",
            description="Small preset challenger with batch_size=4 / grad_accum=4 if VRAM allows.",
            max_train_steps=phase_b_steps,
            overrides=PHASE_A_OVERRIDES + ("--batch_size", "4", "--gradient_accumulation_steps", "4"),
        ),
        "small_phase_b_batch2_accum8": ExperimentSpec(
            name="small_phase_b_batch2_accum8",
            config_name="small",
            description="Fallback small challenger with batch_size=2 / grad_accum=8 to keep effective batch >= 16.",
            max_train_steps=phase_b_steps,
            overrides=PHASE_A_OVERRIDES + ("--batch_size", "2", "--gradient_accumulation_steps", "8"),
        ),
        "middle_phase_c_workers8": ExperimentSpec(
            name="middle_phase_c_workers8",
            config_name="middle",
            description="Phase C option when data_wait_ratio stays high and CPU is not saturated.",
            max_train_steps=benchmark_steps,
            overrides=PHASE_C_CPU_FREE_OVERRIDES,
        ),
        "middle_phase_c_prefetch2_cache32": ExperimentSpec(
            name="middle_phase_c_prefetch2_cache32",
            config_name="middle",
            description="Phase C option when data_wait_ratio stays high and CPU is already saturated.",
            max_train_steps=benchmark_steps,
            overrides=PHASE_C_CPU_BOUND_OVERRIDES,
        ),
    }


ALIASES = {
    "phase-a-sequence": ("middle_baseline", "middle_phase_a"),
    "phase-b-candidates": ("small_phase_b_batch4", "small_phase_b_batch2_accum8"),
    "phase-c-candidates": ("middle_phase_c_workers8", "middle_phase_c_prefetch2_cache32"),
    "all-train": (
        "middle_baseline",
        "middle_phase_a",
        "small_phase_b_batch4",
        "small_phase_b_batch2_accum8",
        "middle_phase_c_workers8",
        "middle_phase_c_prefetch2_cache32",
    ),
}


def _expand_names(requested: list[str], available: dict[str, ExperimentSpec]) -> list[ExperimentSpec]:
    expanded: list[str] = []
    for name in requested:
        if name in available:
            expanded.append(name)
            continue
        alias = ALIASES.get(name)
        if alias is None:
            valid_names = ", ".join(sorted(list(available.keys()) + list(ALIASES.keys())))
            raise SystemExit(f"Unknown experiment '{name}'. Valid names: {valid_names}")
        expanded.extend(alias)

    deduped: list[str] = []
    seen: set[str] = set()
    for name in expanded:
        if name not in seen:
            deduped.append(name)
            seen.add(name)
    return [available[name] for name in deduped]


def _print_commands(
    specs: list[ExperimentSpec],
    *,
    data_dir: Path,
    output_root: Path,
    train_profile: str,
    accelerate_config: Path | None,
    python_executable: Path,
    prepared_cache_dir: Path | None,
    command_view: str,
) -> None:
    warning = _local_storage_warning(data_dir, prepared_cache_dir)
    if warning is not None:
        print(f"# {warning}")
        print()

    for spec in specs:
        print(f"[{spec.name}]")
        print(f"# {spec.description}")
        if command_view in {"train", "both"}:
            train_command = _train_command(
                spec,
                data_dir=data_dir,
                output_root=output_root,
                train_profile=train_profile,
                accelerate_config=accelerate_config,
                python_executable=python_executable,
                prepared_cache_dir=prepared_cache_dir,
            )
            print(_quote_command(train_command))
        if command_view in {"validate", "both"}:
            if command_view == "both":
                print()
            print()
            print(f"# Full offline validation for {spec.name}")
            print(
                _quote_command(
                    _validate_command(
                        spec,
                        data_dir=data_dir,
                        output_root=output_root,
                        accelerate_config=accelerate_config,
                        python_executable=python_executable,
                    )
                )
            )
        print()


def _run_commands(commands: list[list[str]], *, dry_run: bool) -> None:
    for command in commands:
        print(f"$ {_quote_command(command)}")
        if dry_run:
            continue
        subprocess.run(command, check=True, cwd=PROJECT_ROOT)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and execute the 8GB throughput experiment plan using direct main.py launches."
    )
    parser.add_argument(
        "mode",
        choices=["list", "show", "run", "show-validations", "run-validations"],
        help="List available experiments, print commands, or execute them.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        dest="experiments",
        help="Experiment name or alias. Repeatable. Defaults depend on mode.",
    )
    parser.add_argument("--data-dir", type=Path, required=False, help="Dataset root passed to main.py.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "runs" / "throughput_8gb")
    parser.add_argument("--prepared-cache-dir", type=Path, default=None)
    parser.add_argument("--benchmark-steps", type=int, default=500, help="Joint-training steps for middle runs.")
    parser.add_argument("--train-profile", default="auto")
    parser.add_argument(
        "--accelerate-config",
        type=Path,
        default=DEFAULT_ACCELERATE_CONFIG if DEFAULT_ACCELERATE_CONFIG.exists() else None,
        help="Optional accelerate config file passed to accelerate launch.",
    )
    parser.add_argument(
        "--python-executable",
        type=Path,
        default=_default_python_executable(),
        help="Python executable used for the generated accelerate commands.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them.")
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()

    available = _specs(max(1, int(args.benchmark_steps)))
    default_names = {
        "show": ["phase-a-sequence"],
        "run": ["phase-a-sequence"],
        "show-validations": ["phase-a-sequence"],
        "run-validations": ["phase-a-sequence"],
    }

    if args.mode == "list":
        for name, spec in available.items():
            print(f"{name}: {spec.description}")
        for alias, members in sorted(ALIASES.items()):
            print(f"{alias}: {', '.join(members)}")
        return 0

    if args.data_dir is None:
        parser.error("--data-dir is required for this mode.")

    data_dir = args.data_dir.resolve()
    output_root = args.output_root.resolve()
    prepared_cache_dir = args.prepared_cache_dir.resolve() if args.prepared_cache_dir is not None else None
    accelerate_config = args.accelerate_config.resolve() if args.accelerate_config is not None else None
    python_executable = args.python_executable.resolve()
    requested = args.experiments or default_names[args.mode]
    specs = _expand_names(requested, available)

    if args.mode == "show":
        _print_commands(
            specs,
            data_dir=data_dir,
            output_root=output_root,
            train_profile=args.train_profile,
            accelerate_config=accelerate_config,
            python_executable=python_executable,
            prepared_cache_dir=prepared_cache_dir,
            command_view="train",
        )
        return 0

    if args.mode == "show-validations":
        _print_commands(
            specs,
            data_dir=data_dir,
            output_root=output_root,
            train_profile=args.train_profile,
            accelerate_config=accelerate_config,
            python_executable=python_executable,
            prepared_cache_dir=prepared_cache_dir,
            command_view="validate",
        )
        return 0

    if not data_dir.exists():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")

    output_root.mkdir(parents=True, exist_ok=True)

    if args.mode == "run":
        commands = [
            _train_command(
                spec,
                data_dir=data_dir,
                output_root=output_root,
                train_profile=args.train_profile,
                accelerate_config=accelerate_config,
                python_executable=python_executable,
                prepared_cache_dir=prepared_cache_dir,
            )
            for spec in specs
        ]
        _run_commands(commands, dry_run=args.dry_run)
        return 0

    if args.mode == "run-validations":
        commands = [
            _validate_command(
                spec,
                data_dir=data_dir,
                output_root=output_root,
                accelerate_config=accelerate_config,
                python_executable=python_executable,
            )
            for spec in specs
        ]
        _run_commands(commands, dry_run=args.dry_run)
        return 0

    raise SystemExit(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
