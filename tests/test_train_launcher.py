from __future__ import annotations

from utils.train_launcher import build_train_command, build_validate_command, plan_from_env


def test_build_train_command_supports_bool_and_list_overrides():
    command = build_train_command(
        accelerate_bin="/tmp/accelerate",
        config_path="configs/train/small.yaml",
        data_dir="/tmp/data",
        output_dir="runs/demo",
        environ={
            "TRAIN_PROFILE": "auto",
            "USE_RETINEX": "false",
            "ATTENTION_BACKEND": "auto",
            "USE_TORCH_COMPILE": "true",
            "TORCH_COMPILE_MODE": "reduce-overhead",
            "BENCHMARK_INFERENCE_STEPS": "8 20",
            "PREPARE_FORCE": "1",
            "BATCH_SIZE": "6",
        },
    )

    assert command[:10] == (
        "/tmp/accelerate",
        "launch",
        "main.py",
        "--mode",
        "train",
        "--config",
        "configs/train/small.yaml",
        "--data_dir",
        "/tmp/data",
        "--output_dir",
    )
    assert "--no-use_retinex" in command
    assert "--prepare_force" in command
    assert command[command.index("--attention_backend") + 1] == "auto"
    assert "--use_torch_compile" in command
    assert command[command.index("--torch_compile_mode") + 1] == "reduce-overhead"
    assert command[command.index("--train_profile") + 1] == "auto"
    assert command[command.index("--batch_size") + 1] == "6"
    bench_index = command.index("--benchmark_inference_steps")
    assert command[bench_index + 1 : bench_index + 3] == ("8", "20")


def test_build_validate_command_uses_standardized_defaults():
    command = build_validate_command(
        accelerate_bin="/tmp/accelerate",
        config_path="configs/train/middle.yaml",
        data_dir="/tmp/data",
        output_dir="runs/demo",
        environ={},
    )

    assert command[command.index("--model_path") + 1] == "runs/demo/best_model"
    assert command[command.index("--output_dir") + 1] == "runs/demo/full_eval"
    assert command[command.index("--batch_size") + 1] == "2"
    assert command[command.index("--num_validation_images") + 1] == "12"
    bench_index = command.index("--benchmark_inference_steps")
    assert command[bench_index + 1 : bench_index + 3] == ("8", "20")
    assert command[command.index("--semantic_backbone") + 1] == "resnet18"
    assert command[command.index("--nr_metric") + 1] == "niqe"


def test_plan_from_env_builds_train_and_post_eval_commands():
    plan = plan_from_env(
        {
            "ACCELERATE_BIN": "/tmp/accelerate",
            "MODEL_SIZE": "middle",
            "DATA_DIR": "/tmp/data",
            "OUTPUT_DIR": "runs/exp",
            "RUN_FULL_EVAL_AFTER_TRAIN": "true",
        }
    )

    assert plan.run_mode == "train"
    assert plan.config_path == "configs/train/middle.yaml"
    assert plan.primary_command[0] == "/tmp/accelerate"
    assert plan.post_success_command is not None
    assert "--mode" in plan.post_success_command
    assert plan.post_success_command[plan.post_success_command.index("--mode") + 1] == "validate"


def test_plan_from_env_builds_validate_command_directly():
    plan = plan_from_env(
        {
            "ACCELERATE_BIN": "/tmp/accelerate",
            "RUN_MODE": "validate",
            "CONFIG_PATH": "small",
            "DATA_DIR": "/tmp/data",
            "OUTPUT_DIR": "runs/eval",
            "MODEL_PATH": "runs/eval/checkpoint",
        }
    )

    assert plan.run_mode == "validate"
    assert plan.config_path == "configs/train/small.yaml"
    assert plan.post_success_command is None
    assert plan.primary_command[plan.primary_command.index("--model_path") + 1] == "runs/eval/checkpoint"
