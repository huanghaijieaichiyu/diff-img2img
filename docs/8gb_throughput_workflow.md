# 8GB Throughput Workflow

This repo-local workflow implements the 8GB single-GPU throughput plan without touching `start_train.sh`.

## Files

- `scripts/run_8gb_throughput.py`
  - Generates or executes direct `main.py` + `accelerate` commands for:
    - `middle_baseline`
    - `middle_phase_a`
    - `small_phase_b_batch4`
    - `small_phase_b_batch2_accum8`
    - `middle_phase_c_workers8`
    - `middle_phase_c_prefetch2_cache32`
- `scripts/evaluate_8gb_throughput.py`
  - Reads `training_metrics.csv`, `metrics.txt`, and `training_status.json`
  - Applies the plan thresholds for Phase A and Phase B
  - Recommends the next experiment or the default winner

## Typical Usage

Print the baseline + Phase A commands:

```bash
python3 scripts/run_8gb_throughput.py show \
  --data-dir /path/to/dataset
```

Run the baseline + Phase A sequence:

```bash
python3 scripts/run_8gb_throughput.py run \
  --data-dir /path/to/dataset \
  --experiment phase-a-sequence
```

Print the full offline validation commands:

```bash
python3 scripts/run_8gb_throughput.py show-validations \
  --data-dir /path/to/dataset \
  --experiment phase-a-sequence
```

Run the Phase B small candidates after Phase A passes:

```bash
python3 scripts/run_8gb_throughput.py run \
  --data-dir /path/to/dataset \
  --experiment phase-b-candidates
```

Evaluate the results:

```bash
python3 scripts/evaluate_8gb_throughput.py \
  --baseline-run runs/throughput_8gb/middle_baseline \
  --phase-a-run runs/throughput_8gb/middle_phase_a \
  --phase-b-run runs/throughput_8gb/small_phase_b_batch4 \
  --phase-b-run runs/throughput_8gb/small_phase_b_batch2_accum8
```

## Notes

- The runner warns if `data_dir` or `.prepared` live under `/mnt/...`, because cross-mounted storage can skew throughput numbers.
- The full offline validation commands always restore:
  - `semantic_backbone=resnet18`
  - `nr_metric=niqe`
  - `benchmark_inference_steps=[8, 20]`
  - `num_validation_images=12`
- The comparison script prefers `full_eval/metrics.txt` when present and falls back to the training-time `metrics.txt`.
