from __future__ import annotations

import json
import pytest

from utils.run_artifacts import summarize_run_dir


def test_summarize_run_dir_reads_metrics_and_resolved_config(tmp_path):
    run_dir = tmp_path / "run_a"
    run_dir.mkdir()

    (run_dir / "training_metrics.csv").write_text(
        "step,phase,samples_per_sec,data_wait_ratio,cpu_percent,gpu_reserved_gb\n"
        "100,joint,1.1,0.20,50,4.0\n"
        "110,joint,1.3,0.10,60,4.5\n",
        encoding="utf-8",
    )
    (run_dir / "metrics.txt").write_text(
        "[steps=8]\n"
        "PSNR: 20.5000\n"
        "SSIM: 0.8010\n"
        "LPIPS: 0.1200\n"
        "SecondsPerImage: 0.050000\n"
        "Step: 110\n",
        encoding="utf-8",
    )
    (run_dir / "resolved_config.json").write_text(
        json.dumps(
            {
                "summary": {
                    "config_name": "middle",
                    "effective_batch_size": 8,
                    "resolution": 256,
                    "decom_variant": "naf",
                    "condition_variant": "max_v2",
                }
            }
        ),
        encoding="utf-8",
    )

    summary = summarize_run_dir(run_dir, window=2, preferred_step=8)
    assert summary.name == "run_a"
    assert summary.latest_step == 110
    assert summary.psnr == 20.5
    assert summary.throughput_mean == pytest.approx(1.2)
    assert summary.resolved_config["summary"]["config_name"] == "middle"
