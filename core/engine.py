import csv
import json
import logging
import math
import os
import random
import shutil
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm.auto import tqdm

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.schedulers.scheduling_dpmsolver_multistep import DPMSolverMultistepScheduler
from diffusers.training_utils import EMAModel
from torcheval.metrics.functional import peak_signal_noise_ratio

from datasets.data_set import LowLightDataset
from models.conditioning import build_condition_adapter
from models.diffusion import CombinedModel
from models.retinex import build_decom_net
from utils.loss import CompositeLoss
from utils.metrics import SemanticFeatureMetric, try_compute_niqe
from utils.misc import charbonnier_loss_elementwise, compute_min_snr_loss_weights, compute_adaptive_loss_weights, ssim
from utils.project_config import build_runtime_summary, serialize_args
from utils.runtime_backend import (
    resolve_unet_runtime_backend,
    unwrap_compiled_module,
)
from utils.train_display import create_training_display
from utils.video_writer import video_writer

logger = get_logger(__name__, log_level="INFO")

SUPPORTED_VALIDATION_METRICS = ("psnr", "ssim", "lpips", "semantic_distance", "niqe")


def normalize_validation_metrics(metrics) -> tuple[str, ...]:
    if metrics in (None, "", []):
        return ("psnr", "ssim")
    if isinstance(metrics, str):
        metrics = [part.strip() for part in metrics.replace(",", " ").split() if part.strip()]
    normalized = []
    requested = []
    for metric in metrics:
        metric_name = str(metric).strip().lower()
        if metric_name in SUPPORTED_VALIDATION_METRICS and metric_name not in requested:
            requested.append(metric_name)
    for metric_name in ("psnr", "ssim"):
        if metric_name not in normalized:
            normalized.append(metric_name)
    for metric_name in requested:
        if metric_name not in normalized:
            normalized.append(metric_name)
    return tuple(normalized)


def resolve_validation_step_counts(
    num_inference_steps: int,
    benchmark_inference_steps,
    train_validation_benchmark_steps,
    *,
    fast: bool,
) -> list[int]:
    raw_steps = train_validation_benchmark_steps if fast else benchmark_inference_steps
    if raw_steps in (None, "", []):
        raw_steps = [num_inference_steps]
    ordered_steps = []
    for step_count in list(raw_steps) + [num_inference_steps]:
        value = int(step_count)
        if value > 0 and value not in ordered_steps:
            ordered_steps.append(value)
    return ordered_steps


class DiffusionEngine:
    def __init__(self, args):
        self.args = args
        self.best_psnr = -1.0
        self.csv_fields = [
            "step",
            "epoch",
            "phase",
            "loss",
            "lr",
            "l_diff",
            "l_x0",
            "l_wavelet",
            "x0_w",
            "l_ret",
            "l_recon_low",
            "l_recon_high",
            "l_consistency",
            "l_exposure",
            "l_tv",
            "data_time",
            "compute_time",
            "iter_time",
            "data_wait_ratio",
            "samples_per_sec",
            "cpu_percent",
            "cpu_rss_gb",
            "gpu_allocated_gb",
            "gpu_reserved_gb",
            "gpu_max_reserved_gb",
            "val_psnr",
            "val_ssim",
            "val_lpips",
            "val_seconds_per_image",
            "val_step_count",
        ]
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        self.latest_validation_metrics = {}
        self.status_json_path = os.path.join(args.output_dir, "training_status.json")
        self.resolved_config_path = os.path.join(args.output_dir, "resolved_config.json")
        self.runtime_summary = build_runtime_summary(args)
        self._eval_dataloader = None
        self._lpips_fn = None
        self._lpips_load_attempted = False
        self._semantic_metric = None

        logging_dir = os.path.join(args.output_dir, "logs")
        accelerator_project_config = ProjectConfiguration(
            project_dir=args.output_dir,
            logging_dir=logging_dir,
        )
        self.accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_config=accelerator_project_config,
        )

        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(self.accelerator.state, main_process_only=False)

        if args.seed is not None:
            set_seed(args.seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")

        if self.accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
            logger.info(
                "GPU memory note: gpu_allocated_gb tracks active tensor memory, while gpu_reserved_gb tracks the allocator cache pool."
            )

        self.metrics_csv_path = os.path.join(args.output_dir, "training_metrics.csv")
        self._joint_decom_trainable = None
        self.unet_runtime_backend = None
        self._setup_models()

        if self.accelerator.is_main_process:
            self._write_resolved_config()

        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="squaredcos_cap_v2",
            prediction_type=args.prediction_type,
        )
        use_uncertainty = getattr(args, "use_uncertainty_weighting", False)
        self.criterion = CompositeLoss(
            device=self.accelerator.device,
            use_uncertainty_weighting=use_uncertainty,
            use_lpips=getattr(args, "use_lpips", True),
            lpips_resize=getattr(args, "lpips_resize", None),
            w_wavelet=getattr(args, "wavelet_loss_weight", 0.0),
        ).to(self.accelerator.device)

    def _validation_plan(self, fast: bool) -> tuple[list[int], tuple[str, ...]]:
        if fast:
            metric_names = normalize_validation_metrics(getattr(self.args, "train_validation_metrics", ("psnr", "ssim")))
        else:
            metric_names = ("psnr", "ssim", "lpips", "semantic_distance", "niqe")
        step_counts = resolve_validation_step_counts(
            self.args.num_inference_steps,
            self.args.benchmark_inference_steps,
            getattr(self.args, "train_validation_benchmark_steps", None),
            fast=fast,
        )
        return step_counts, metric_names

    def _get_eval_dataloader(self):
        if self._eval_dataloader is not None:
            return self._eval_dataloader

        eval_dataset = LowLightDataset(
            image_dir=self.args.data_dir,
            img_size=self.args.resolution,
            phase="test",
            decode_cache_size=getattr(self.args, "decode_cache_size", 0),
        )
        eval_loader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": False,
            "num_workers": self.args.num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", True)),
            "worker_init_fn": self._worker_init_fn,
        }
        if self.args.num_workers > 0:
            eval_loader_kwargs.update({
                "persistent_workers": bool(getattr(self.args, "persistent_workers", True)),
                "prefetch_factor": max(2, int(getattr(self.args, "prefetch_factor", 4))),
            })
        eval_dataloader = DataLoader(eval_dataset, **eval_loader_kwargs)
        self._eval_dataloader = self.accelerator.prepare(eval_dataloader)
        return self._eval_dataloader

    def _get_lpips_fn(self):
        if self._lpips_fn is not None:
            return self._lpips_fn
        if self._lpips_load_attempted:
            return None
        self._lpips_load_attempted = True
        try:
            import lpips

            self._lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(self.accelerator.device)
            self._lpips_fn.eval()
        except ImportError:
            logger.warning("lpips not installed. Skipping LPIPS metric.")
            self._lpips_fn = None
        return self._lpips_fn

    def _get_semantic_metric(self):
        if self._semantic_metric is None:
            self._semantic_metric = SemanticFeatureMetric(
                self.accelerator.device,
                backbone=self.args.semantic_backbone,
            )
        return self._semantic_metric

    def _setup_models(self):
        # Dynamic channel calculation based on preset
        cond_channels = getattr(self.args, 'cond_out_channels', 7)
        input_channels = 3 + cond_channels  # noisy + condition_map
        logger.info(f"Initializing UNet (Input Channels={input_channels}, Condition Channels={cond_channels})...")
        self.unet = UNet2DModel(
            sample_size=self.args.resolution,
            in_channels=input_channels,
            out_channels=3,
            layers_per_block=self.args.unet_layers_per_block,
            block_out_channels=self.args.unet_block_channels,
            down_block_types=tuple(self.args.unet_down_block_types),
            up_block_types=tuple(self.args.unet_up_block_types),
        )

        self.decom_model = None
        if self.args.use_retinex:
            logger.info("Initializing Retinex decomposition network...")
            self.decom_model = build_decom_net(
                variant=self.args.decom_variant,
                base_channel=self.args.decom_base_channels,
            ).to(self.accelerator.device)

        self.condition_adapter = build_condition_adapter(
            variant=self.args.condition_variant,
            block_channels=self.args.unet_block_channels,
            cond_out_channels=cond_channels,
            base_channels=self.args.base_condition_channels,
            use_retinex=self.args.use_retinex,
            conditioning_space=self.args.conditioning_space,
        ).to(self.accelerator.device)

        if self.args.model_path:
            self._load_checkpoint(self.args.model_path)

        self.unet_runtime_backend = self._configure_unet_runtime_backend()

        if self.args.use_retinex and self.args.decom_warmup_steps > 0:
            self._set_trainable(self.unet, False)
            self._set_trainable(self.condition_adapter, False)

        self.training_model = CombinedModel(
            self.unet,
            self.decom_model,
            self.condition_adapter,
            conditioning_space=self.args.conditioning_space,
            inject_mode=getattr(self.args, "inject_mode", "concat_pyramid"),
        )

        self.ema_models = {}
        if self.args.use_ema:
            logger.info("Initializing EMA models...")
            self.ema_models["unet"] = EMAModel(
                self.unet.parameters(),
                decay=self.args.ema_decay,
                model_cls=UNet2DModel,
                model_config=unwrap_compiled_module(self.unet).config,
            )
            if self.decom_model is not None:
                self.ema_models["decom_model"] = EMAModel(
                    self.decom_model.parameters(),
                    decay=self.args.ema_decay,
                )
            if self.condition_adapter is not None:
                self.ema_models["condition_adapter"] = EMAModel(
                    self.condition_adapter.parameters(),
                    decay=self.args.ema_decay,
                )
            for ema_model in self.ema_models.values():
                ema_model.to(self.accelerator.device)

    def _record_unet_backend_summary(self, backend):
        if backend is None:
            return
        payload = backend.as_dict()
        for key, value in payload.items():
            self.runtime_summary[f"unet_backend_{key}"] = value

    def _configure_unet_runtime_backend(self):
        backend = resolve_unet_runtime_backend(self.args)
        checkpointing_enabled = bool(getattr(self.args, "enable_gradient_checkpointing", False))

        if checkpointing_enabled:
            self.unet.enable_gradient_checkpointing()
            if self.accelerator.is_main_process:
                logger.info("Enabled gradient checkpointing on UNet")

        try:
            backend = self._apply_unet_runtime_backend(backend)
        finally:
            self._record_unet_backend_summary(backend)

        if self.accelerator.is_main_process:
            reason_text = "; ".join(backend.reasons) if backend.reasons else "no extra notes"
            logger.info(
                "UNet runtime backend: requested=%s resolved=%s compile=%s xformers=%s gradient_checkpointing=%s (%s)",
                backend.requested_backend,
                backend.resolved_backend,
                backend.compile_enabled,
                backend.xformers_enabled,
                checkpointing_enabled,
                reason_text,
            )
        return backend

    def _apply_unet_runtime_backend(self, backend):
        if backend.compile_enabled:
            try:
                self.unet.compile(mode=backend.torch_compile_mode)
            except Exception as exc:
                if backend.requested_backend == "compile":
                    raise
                fallback_backend = "xformers" if backend.xformers_requested else "native"
                backend = replace(
                    backend,
                    resolved_backend=fallback_backend,
                    compile_enabled=False,
                    xformers_enabled=fallback_backend == "xformers",
                ).with_reason(f"torch.compile failed and auto backend fell back to {fallback_backend}: {exc}")
            else:
                return backend.with_reason("Applied torch.compile to the UNet in-place.")

        if backend.xformers_enabled:
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as exc:
                if backend.requested_backend == "xformers":
                    raise
                backend = replace(
                    backend,
                    resolved_backend="native",
                    xformers_enabled=False,
                ).with_reason(f"xformers enable failed and backend fell back to native attention: {exc}")
            else:
                return backend.with_reason("Enabled xformers memory efficient attention.")

        return backend

    @staticmethod
    def _set_trainable(module, trainable: bool):
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = trainable

    @staticmethod
    def _unwrap_compiled_module(module):
        return unwrap_compiled_module(module)

    @staticmethod
    def _parallel_wrapper_types():
        wrapper_types = [torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel]
        try:
            from deepspeed import DeepSpeedEngine

            wrapper_types.append(DeepSpeedEngine)
        except Exception:
            pass
        try:
            from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

            wrapper_types.append(FSDP)
        except Exception:
            pass
        return tuple(wrapper_types)

    def _unwrap_training_model(self, model=None):
        model = self.training_model if model is None else model
        model = self._unwrap_compiled_module(model)
        wrapper_types = self._parallel_wrapper_types()
        while isinstance(model, wrapper_types):
            model = model.module
            model = self._unwrap_compiled_module(model)
        return model

    def _named_core_modules(self, unwrapped_model=None):
        if unwrapped_model is None:
            unwrapped_model = self._unwrap_training_model()
        return {
            "unet": unwrapped_model.unet,
            "decom_model": unwrapped_model.decom_model,
            "condition_adapter": unwrapped_model.condition_adapter,
        }

    @staticmethod
    def _use_weight_decay(parameter_name: str, parameter: torch.nn.Parameter) -> bool:
        normalized_name = parameter_name.lower()
        if parameter.ndim <= 1:
            return False
        if normalized_name.endswith("bias"):
            return False
        if any(token in normalized_name for token in ("norm", "bn", "log_vars")):
            return False
        return True

    def _build_optimizer_param_groups(self):
        decay_params = []
        no_decay_params = []
        seen = set()

        def add_named_params(named_params, include_frozen: bool):
            for name, parameter in named_params:
                if not include_frozen and not parameter.requires_grad:
                    continue
                param_id = id(parameter)
                if param_id in seen:
                    continue
                seen.add(param_id)
                if self._use_weight_decay(name, parameter):
                    decay_params.append(parameter)
                else:
                    no_decay_params.append(parameter)

        add_named_params(self.training_model.named_parameters(), include_frozen=True)
        add_named_params(self.criterion.named_parameters(), include_frozen=False)

        param_groups = []
        if decay_params:
            param_groups.append({"params": decay_params, "weight_decay": 0.01})
        if no_decay_params:
            param_groups.append({"params": no_decay_params, "weight_decay": 0.0})
        return param_groups

    def _params_for_gradient_clipping(self):
        params = [parameter for parameter in self.training_model.parameters() if parameter.requires_grad]
        params.extend(parameter for parameter in self.criterion.parameters() if parameter.requires_grad)
        return params

    @staticmethod
    def _optimizer_has_gradients(optimizer) -> bool:
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    return True
        return False

    def _stage_boundaries(self) -> tuple[int, int]:
        stage1_end = getattr(self.args, "stage1_decom_only_steps", 0)
        stage2_end = stage1_end + getattr(self.args, "stage2_adapter_steps", 0)
        return stage1_end, stage2_end

    def _set_training_stage(self, global_step: int):
        """
        Three-stage training strategy:
        Stage 1 (0-stage1_end): Retinex decomposition only
        Stage 2 (stage1_end-stage2_end): Decomposition + Adapter (UNet frozen)
        Stage 3 (stage2_end+): Full joint training with gradual decom unfreezing
        """
        unwrapped = self._unwrap_training_model()

        stage1_end, stage2_end = self._stage_boundaries()

        if global_step < stage1_end:
            # Stage 1: Only decomposition
            self._set_trainable(unwrapped.decom_model, True)
            self._set_trainable(unwrapped.condition_adapter, False)
            self._set_trainable(unwrapped.unet, False)
            return "stage1_decom"

        elif global_step < stage2_end:
            # Stage 2: Decomposition + Adapter
            self._set_trainable(unwrapped.decom_model, True)
            self._set_trainable(unwrapped.condition_adapter, True)
            self._set_trainable(unwrapped.unet, False)
            return "stage2_adapter"

        else:
            # Stage 3: Full joint training
            joint_step = global_step - stage2_end
            self._set_trainable(unwrapped.unet, True)
            self._set_trainable(unwrapped.condition_adapter, True)

            # Gradual decom unfreezing
            decom_trainable = joint_step >= self.args.freeze_decom_steps
            if self._joint_decom_trainable != decom_trainable:
                self._set_trainable(unwrapped.decom_model, decom_trainable)
                self._joint_decom_trainable = decom_trainable
                state = "trainable" if decom_trainable else "frozen"
                logger.info(f"Setting decomposition network to {state} at joint_step={joint_step}")

            return "stage3_joint"

    def _set_joint_phase_trainability(self, joint_step: int):
        """Legacy method for backward compatibility"""
        unwrapped_model = self._unwrap_training_model()
        self._set_trainable(unwrapped_model.unet, True)
        self._set_trainable(unwrapped_model.condition_adapter, True)

        if unwrapped_model.decom_model is None:
            self._joint_decom_trainable = None
            return

        decom_trainable = joint_step >= self.args.freeze_decom_steps
        if self._joint_decom_trainable == decom_trainable:
            return

        self._set_trainable(unwrapped_model.decom_model, decom_trainable)
        self._joint_decom_trainable = decom_trainable
        state = "trainable" if decom_trainable else "frozen"
        logger.info(f"Setting decomposition network to {state} at joint_step={joint_step}")

    def _ema_file_map(self, save_path: str):
        return {
            "unet": [
                os.path.join(save_path, "ema_unet.pth"),
                os.path.join(save_path, "ema_model.pth"),
            ],
            "decom_model": [os.path.join(save_path, "ema_decom_model.pth")],
            "condition_adapter": [os.path.join(save_path, "ema_condition_adapter.pth")],
        }

    def _load_ema_state(self, save_path: str):
        if not self.args.use_ema or not self.ema_models:
            return

        for module_name, candidate_paths in self._ema_file_map(save_path).items():
            ema_model = self.ema_models.get(module_name)
            if ema_model is None:
                continue
            for candidate_path in candidate_paths:
                if not os.path.exists(candidate_path):
                    continue
                ema_model.load_state_dict(torch.load(candidate_path, map_location=self.accelerator.device))
                logger.info(f"Loaded EMA state for {module_name} from {candidate_path}")
                break

    def _save_ema_state(self, save_path: str):
        if not self.args.use_ema or not self.ema_models:
            return

        file_map = self._ema_file_map(save_path)
        for module_name, ema_model in self.ema_models.items():
            target_path = file_map[module_name][0]
            torch.save(ema_model.state_dict(), target_path)
            if module_name == "unet":
                torch.save(ema_model.state_dict(), os.path.join(save_path, "ema_model.pth"))

    def _ema_store(self):
        if not self.args.use_ema or not self.ema_models:
            return

        modules = self._named_core_modules()
        for module_name, ema_model in self.ema_models.items():
            module = modules.get(module_name)
            if module is not None:
                ema_model.store(module.parameters())

    def _ema_copy_to_modules(self):
        if not self.args.use_ema or not self.ema_models:
            return

        modules = self._named_core_modules()
        for module_name, ema_model in self.ema_models.items():
            module = modules.get(module_name)
            if module is not None:
                ema_model.copy_to(module.parameters())

    def _ema_restore(self):
        if not self.args.use_ema or not self.ema_models:
            return

        modules = self._named_core_modules()
        for module_name, ema_model in self.ema_models.items():
            module = modules.get(module_name)
            if module is not None:
                ema_model.restore(module.parameters())

    def _ema_step(self):
        """
        P1 Fix: Adaptive EMA decay with warmup for better early training stability.
        Starts with lower decay (0.95) and gradually increases to target (0.9999).
        """
        if not self.args.use_ema or not self.ema_models:
            return

        # Calculate adaptive decay based on training progress
        warmup_steps = getattr(self.args, 'ema_warmup_steps', 5000)
        # Track global step through accelerator
        current_step = getattr(self, '_global_step', 0)

        if current_step < warmup_steps:
            # Cosine warmup from 0.95 to target decay
            progress = current_step / warmup_steps
            decay_start = 0.95
            decay_end = self.args.ema_decay
            # Smooth cosine interpolation
            decay = decay_start + (decay_end - decay_start) * (0.5 * (1 - math.cos(math.pi * progress)))
        else:
            decay = self.args.ema_decay

        # Update all EMA models with adaptive decay
        modules = self._named_core_modules()
        for module_name, ema_model in self.ema_models.items():
            module = modules.get(module_name)
            if module is not None:
                ema_model.decay = decay
                ema_model.step(module.parameters())

    def _candidate_paths(self, path: str, filename: str):
        return [
            os.path.join(path, "unet_final", filename),
            os.path.join(path, filename),
            os.path.join(os.path.dirname(path), filename),
        ]

    def _load_checkpoint(self, path):
        logger.info(f"Loading model from {path}...")

        unet_path = os.path.join(path, "unet_final")
        if not os.path.exists(unet_path):
            unet_path = os.path.join(path, "unet_best")
        if not os.path.exists(unet_path):
            unet_path = path

        try:
            self.unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=True)
        except Exception:
            try:
                self.unet = UNet2DModel.from_pretrained(unet_path, use_safetensors=False)
            except Exception as exc:
                logger.warning(f"Could not load UNet from {unet_path}: {exc}")

        self.unet.to(self.accelerator.device)

        if self.decom_model is not None:
            loaded = False
            for decom_path in self._candidate_paths(path, "decom_model.pth") + self._candidate_paths(path, "decom_model_best.pth"):
                if os.path.exists(decom_path):
                    try:
                        self.decom_model.load_state_dict(torch.load(decom_path, map_location=self.accelerator.device))
                        logger.info(f"DecomNet loaded from {decom_path}.")
                        loaded = True
                        break
                    except Exception as exc:
                        logger.warning(f"Failed to load DecomNet from {decom_path}: {exc}")
            if not loaded:
                logger.warning("DecomNet weights not found.")

        loaded_adapter = False
        adapter_candidates = (
            self._candidate_paths(path, "condition_adapter.pth") +
            self._candidate_paths(path, "condition_adapter_best.pth") +
            self._candidate_paths(path, "condition_adapter_final.pth")
        )
        for adapter_path in adapter_candidates:
            if os.path.exists(adapter_path):
                try:
                    self.condition_adapter.load_state_dict(
                        torch.load(adapter_path, map_location=self.accelerator.device),
                        strict=False,
                    )
                    logger.info(f"Condition adapter loaded from {adapter_path}.")
                    loaded_adapter = True
                    break
                except Exception as exc:
                    logger.warning(f"Failed to load condition adapter from {adapter_path}: {exc}")
        if not loaded_adapter:
            logger.warning("Condition adapter weights not found.")

    def _worker_init_fn(self, worker_id: int):
        worker_info = get_worker_info()
        worker_seed = worker_info.seed if worker_info is not None else (self.args.seed or 42) + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed % (2 ** 32 - 1))
        cv2.setNumThreads(max(0, int(getattr(self.args, "opencv_threads_per_worker", 1))))

    def _collect_runtime_metrics(self, data_time: float, compute_time: float, batch_size: int) -> dict:
        iter_time = max(data_time + compute_time, 1e-8)
        metrics = {
            "data_time": data_time,
            "compute_time": compute_time,
            "iter_time": iter_time,
            "data_wait_ratio": data_time / iter_time,
            "samples_per_sec": (batch_size * max(1, self.accelerator.num_processes)) / iter_time,
            "cpu_percent": self.process.cpu_percent(None),
            "cpu_rss_gb": self.process.memory_info().rss / (1024 ** 3),
        }

        if torch.cuda.is_available():
            device = self.accelerator.device
            metrics.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated(device) / (1024 ** 3),
                "gpu_reserved_gb": torch.cuda.memory_reserved(device) / (1024 ** 3),
                "gpu_max_reserved_gb": torch.cuda.max_memory_reserved(device) / (1024 ** 3),
            })
        else:
            metrics.update({
                "gpu_allocated_gb": 0.0,
                "gpu_reserved_gb": 0.0,
                "gpu_max_reserved_gb": 0.0,
            })
        return metrics

    def _write_status_snapshot(self, logs: dict):
        if not self.accelerator.is_main_process:
            return

        payload = dict(logs)
        payload.update({
            "train_profile": self.args.train_profile,
            "mixed_precision": self.args.mixed_precision,
            "use_retinex": self.args.use_retinex,
            "config_summary": self.runtime_summary,
        })
        payload.update(self.latest_validation_metrics)

        with open(self.status_json_path, "w") as status_file:
            json.dump(payload, status_file, indent=2)

    def _write_resolved_config(self):
        payload = {
            "args": serialize_args(self.args),
            "summary": self.runtime_summary,
        }
        with open(self.resolved_config_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)

    def _append_training_metrics(self, step: int, phase: str, logs: dict):
        if not self.accelerator.is_main_process:
            return

        row = {key: logs.get(key, "") for key in self.csv_fields}
        row["step"] = step
        row["phase"] = phase

        file_exists = os.path.exists(self.metrics_csv_path)
        with open(self.metrics_csv_path, "a", newline="") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.csv_fields)
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
        self._write_status_snapshot(row)

    def _should_log_step(self, step: int, terminal_step: int | None = None) -> bool:
        interval = max(1, int(getattr(self.args, "log_interval", 10)))
        if step <= 1:
            return True
        if terminal_step is not None and step >= terminal_step:
            return True
        return step % interval == 0

    def _x0_branch_weight(self, global_step: int, timesteps: torch.Tensor = None):
        """
        Improved x0 loss scheduling with:
        1. Cosine warmup (existing)
        2. Time-dependent decay (new) - reduce weight for high-noise timesteps
        """
        _, stage2_end = self._stage_boundaries()

        if global_step < stage2_end:
            return 0.0 if timesteps is None else torch.zeros_like(timesteps, dtype=torch.float32)

        if self.args.x0_loss_warmup_steps <= 0:
            base_weight = self.args.x0_loss_weight
        else:
            progress = (global_step - stage2_end) / float(self.args.x0_loss_warmup_steps)
            progress = max(0.0, min(1.0, progress))
            # Cosine warmup: slow start, fast finish
            cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
            base_weight = self.args.x0_loss_weight * cosine_progress

        # Time-dependent decay (new)
        if timesteps is not None:
            t_ratio = timesteps.float() / self.args.x0_loss_t_max
            t_ratio = t_ratio.clamp(0, 1)
            # Cosine decay: full weight at t=0, reduced at t=t_max
            time_decay = 0.5 * (1 + torch.cos(math.pi * t_ratio))
            return base_weight * time_decay

        return base_weight

    def _retinex_branch_multiplier(self, phase: str, joint_step: int) -> float:
        if not self.args.use_retinex:
            return 0.0
        if phase in {"stage1_decom", "decom_warmup", "stage2_adapter"}:
            return float(self.args.retinex_loss_weight)
        if phase != "stage3_joint":
            return float(self.args.retinex_loss_weight)
        if joint_step < self.args.joint_retinex_ramp_steps:
            ramp_progress = joint_step / max(1, self.args.joint_retinex_ramp_steps)
            ramp_weight = 0.5 * (1 - math.cos(math.pi * ramp_progress))
            return float(self.args.retinex_loss_weight) * float(ramp_weight)
        return float(self.args.retinex_loss_weight)

    def _residual_target(self, clean_images: torch.Tensor, low_light_images: torch.Tensor) -> torch.Tensor:
        return (clean_images - low_light_images) * self.args.residual_scale

    def _decode_residual(self, low_light_images: torch.Tensor, residual_pred: torch.Tensor) -> torch.Tensor:
        residual = residual_pred / max(self.args.residual_scale, 1e-6)
        return (low_light_images + residual).clamp(-1, 1)

    def _reconstruct_x0(self, noisy_target: torch.Tensor, model_pred: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(noisy_target.device)
        alpha_prod_t = alphas_cumprod[timesteps][:, None, None, None]
        beta_prod_t = 1 - alpha_prod_t

        if self.args.prediction_type == "epsilon":
            pred_x0 = (noisy_target - beta_prod_t.sqrt() * model_pred) / (alpha_prod_t.sqrt() + 1e-8)
        else:
            pred_x0 = alpha_prod_t.sqrt() * noisy_target - beta_prod_t.sqrt() * model_pred
        return pred_x0

    @staticmethod
    def _tv_loss(image: torch.Tensor) -> torch.Tensor:
        return (
            torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])) +
            torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]))
        )

    def _compute_retinex_losses(self, aux: dict):
        device = self.accelerator.device
        zero = torch.tensor(0.0, device=device)

        if not self.args.use_retinex or aux.get("r_low") is None or aux.get("i_low") is None:
            return {
                "retinex_total": zero,
                "l_recon_low": zero,
                "l_recon_high": zero,
                "l_consistency": zero,
                "l_exposure": zero,
                "l_tv": zero,
            }

        low_light_01 = aux["low_light_01"]
        r_low = aux["r_low"]
        i_low = aux["i_low"]
        recon_low = F.l1_loss(r_low * i_low, low_light_01)

        r_high = aux.get("r_high")
        i_high = aux.get("i_high")
        clean_01 = aux.get("clean_images_01")

        if r_high is None or i_high is None or clean_01 is None:
            return {
                "retinex_total": recon_low,
                "l_recon_low": recon_low,
                "l_recon_high": zero,
                "l_consistency": zero,
                "l_exposure": zero,
                "l_tv": self._tv_loss(i_low),
            }

        recon_high = F.l1_loss(r_high * i_high, clean_01)
        consistency = F.l1_loss(r_low, r_high)
        tv_loss = self._tv_loss(i_low) + self._tv_loss(i_high)

        mean_i_low = i_low.mean(dim=(1, 2, 3))
        mean_i_high = i_high.mean(dim=(1, 2, 3))
        exposure = F.relu(0.05 - (mean_i_high - mean_i_low)).mean()

        total = (
            recon_low +
            recon_high +
            self.args.retinex_consistency_weight * consistency +
            self.args.tv_loss_weight * tv_loss +
            self.args.retinex_exposure_weight * exposure
        )

        return {
            "retinex_total": total,
            "l_recon_low": recon_low,
            "l_recon_high": recon_high,
            "l_consistency": consistency,
            "l_exposure": exposure,
            "l_tv": tv_loss,
        }

    def _build_optimizer_with_grouped_lr(self):
        """
        Create optimizer with component-specific learning rates:
        - UNet: base LR (1e-4)
        - Retinex: 0.5x base LR (more stable)
        - Adapter: 2x base LR (learns faster)
        """
        unwrapped = self._unwrap_training_model()
        base_lr = self.args.lr

        # Separate parameters by component
        unet_params = []
        decom_params = []
        adapter_params = []
        criterion_params = []

        for name, param in unwrapped.named_parameters():
            if 'unet' in name:
                unet_params.append(param)
            elif 'decom_model' in name:
                decom_params.append(param)
            elif 'condition_adapter' in name:
                adapter_params.append(param)

        # Add criterion parameters (uncertainty weights)
        for param in self.criterion.parameters():
            if param.requires_grad:
                criterion_params.append(param)

        # Build parameter groups with weight decay separation
        param_groups = []

        def add_group(params, lr_mult, name):
            decay = [p for p in params if p.ndim > 1]
            no_decay = [p for p in params if p.ndim <= 1]

            if decay:
                param_groups.append({
                    'params': decay,
                    'lr': base_lr * lr_mult,
                    'weight_decay': 0.01,
                })
            if no_decay:
                param_groups.append({
                    'params': no_decay,
                    'lr': base_lr * lr_mult,
                    'weight_decay': 0.0,
                })

        add_group(unet_params, 1.0, 'unet')
        add_group(decom_params, 0.5, 'decom')
        add_group(adapter_params, 2.0, 'adapter')
        add_group(criterion_params, 1.0, 'criterion')

        return torch.optim.AdamW(param_groups, betas=(0.9, 0.999), eps=1e-8)

    def _load_checkpoint_metadata(self, checkpoint_dir: str):
        metadata_path = os.path.join(checkpoint_dir, "checkpoint_metadata.json")
        if not os.path.exists(metadata_path):
            return

        try:
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                metadata = json.load(metadata_file)
        except Exception as exc:
            logger.warning(f"Failed to read checkpoint metadata from {metadata_path}: {exc}")
            return

        best_psnr = metadata.get("best_psnr")
        if isinstance(best_psnr, (int, float)):
            self.best_psnr = float(best_psnr)

    def train(self):
        if getattr(self.args, 'use_grouped_lr', False):
            optimizer = self._build_optimizer_with_grouped_lr()
        else:
            optimizer = torch.optim.AdamW(
                self._build_optimizer_param_groups(),
                lr=self.args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        train_dataset = LowLightDataset(
            image_dir=self.args.data_dir,
            img_size=self.args.resolution,
            phase="train",
            manifest_path=getattr(self.args, "train_manifest_path", None),
            decode_cache_size=getattr(self.args, "decode_cache_size", 0),
            prepared_cache_dir=getattr(self.args, "prepared_cache_dir", None),
        )
        train_loader_kwargs = {
            "batch_size": self.args.batch_size,
            "shuffle": True,
            "num_workers": self.args.num_workers,
            "pin_memory": bool(getattr(self.args, "pin_memory", True)),
            "drop_last": True,
            "worker_init_fn": self._worker_init_fn,
        }
        if self.args.num_workers > 0:
            train_loader_kwargs["persistent_workers"] = bool(getattr(self.args, "persistent_workers", True))
            train_loader_kwargs["prefetch_factor"] = max(2, int(getattr(self.args, "prefetch_factor", 4)))
        train_dataloader = DataLoader(train_dataset, **train_loader_kwargs)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.args.gradient_accumulation_steps)
        if self.args.max_train_steps is None:
            self.args.max_train_steps = self.args.epochs * num_update_steps_per_epoch

        warmup_steps = self.args.decom_warmup_steps if self.args.use_retinex and self.decom_model is not None else 0
        total_training_steps = self.args.max_train_steps + warmup_steps

        # P0 Improvement: Use cosine annealing with warmup for better convergence
        if self.args.lr_scheduler == "cosine_with_warmup":
            from diffusers.optimization import get_cosine_schedule_with_warmup
            lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
                num_training_steps=total_training_steps * self.args.gradient_accumulation_steps,
                num_cycles=0.5,
            )
        else:
            lr_scheduler = get_scheduler(
                self.args.lr_scheduler,
                optimizer=optimizer,
                num_warmup_steps=self.args.lr_warmup_steps * self.args.gradient_accumulation_steps,
                num_training_steps=total_training_steps * self.args.gradient_accumulation_steps,
            )

        if self.args.use_uncertainty_weighting:
            self.training_model, self.criterion, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.training_model,
                self.criterion,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )
        else:
            self.training_model, optimizer, train_dataloader, lr_scheduler = self.accelerator.prepare(
                self.training_model,
                optimizer,
                train_dataloader,
                lr_scheduler,
            )

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(self.args))
            for key, value in list(tracker_config.items()):
                if not isinstance(value, (int, float, str, bool, torch.Tensor)):
                    if isinstance(value, list):
                        tracker_config[key] = str(value)
                    elif value is None:
                        tracker_config[key] = "None"
                    else:
                        del tracker_config[key]
            self.accelerator.init_trackers(Path(self.args.output_dir).name, config=tracker_config)

        global_step = 0
        first_epoch = 0
        completed_joint_steps = 0
        if self.args.resume:
            if self.args.resume == "latest":
                checkpoints = [d for d in os.listdir(self.args.output_dir) if d.startswith("checkpoint-")]
                if checkpoints:
                    checkpoints = sorted(checkpoints, key=lambda item: int(item.split("-")[1]))
                    self.args.resume = os.path.join(self.args.output_dir, checkpoints[-1])
                else:
                    self.args.resume = None

            if self.args.resume and os.path.isdir(self.args.resume):
                logger.info(f"Resuming training from checkpoint: {self.args.resume}")
                self.accelerator.load_state(self.args.resume)
                global_step = int(os.path.basename(self.args.resume).split("-")[1])
                completed_joint_steps = max(0, global_step - warmup_steps)
                first_epoch = completed_joint_steps // max(1, num_update_steps_per_epoch)
                self._load_ema_state(self.args.resume)
                self._load_checkpoint_metadata(self.args.resume)

        if global_step >= warmup_steps:
            self._set_training_stage(completed_joint_steps)

        logger.info("***** Running training *****")
        logger.info(
            "Effective batch size: %s (batch_size=%s x grad_accum=%s)",
            getattr(self.args, "effective_batch_size", self.args.batch_size * self.args.gradient_accumulation_steps),
            self.args.batch_size,
            self.args.gradient_accumulation_steps,
        )
        rich_enabled = self.accelerator.is_local_main_process and sys.stdout.isatty()
        use_tqdm = False
        progress_bar = tqdm(
            range(global_step, total_training_steps),
            initial=global_step,
            total=total_training_steps,
            disable=not use_tqdm,
        )
        live_display = create_training_display(total_steps=total_training_steps, enabled=rich_enabled)
        live_display.start()

        # Track global step for EMA warmup
        self._global_step = global_step
        self.training_model.train()
        train_iterator = iter(train_dataloader)
        last_iter_end = time.perf_counter()
        accum_data_time = 0.0
        accum_compute_time = 0.0
        accum_samples = 0
        while global_step < warmup_steps:
            try:
                batch = next(train_iterator)
            except StopIteration:
                train_iterator = iter(train_dataloader)
                batch = next(train_iterator)

            data_time = time.perf_counter() - last_iter_end
            compute_start = time.perf_counter()
            _, logs = self._train_step(
                batch,
                optimizer,
                lr_scheduler,
                global_step=global_step,
                phase="decom_warmup",
                joint_step=0,
            )
            compute_time = time.perf_counter() - compute_start
            batch_size = batch[0].shape[0]
            accum_data_time += data_time
            accum_compute_time += compute_time
            accum_samples += batch_size

            if self.accelerator.sync_gradients:
                runtime_metrics = self._collect_runtime_metrics(
                    data_time=accum_data_time,
                    compute_time=accum_compute_time,
                    batch_size=accum_samples,
                )
                logs.update(runtime_metrics)
                logs.update(self.latest_validation_metrics)
                progress_bar.update(1)
                global_step += 1
                self._global_step = global_step  # Update for EMA warmup
                logs["epoch"] = 0
                logs["step"] = global_step
                logs["phase"] = "decom_warmup"
                numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                if self.accelerator.is_main_process and self._should_log_step(global_step, terminal_step=warmup_steps):
                    live_display.update(logs)
                    if not rich_enabled:
                        logger.info(
                            f"[warmup] step={global_step}/{total_training_steps} "
                            f"loss={logs['loss']:.4f} l_ret={logs['l_ret']:.4f} "
                            f"samples/s={logs['samples_per_sec']:.2f} "
                            f"data={logs['data_time']:.3f}s compute={logs['compute_time']:.3f}s "
                            f"cpu={logs['cpu_percent']:.1f}% gpu_alloc={logs['gpu_allocated_gb']:.2f}GB "
                            f"gpu_resv={logs['gpu_reserved_gb']:.2f}GB"
                        )
                    self.accelerator.log(numeric_logs, step=global_step)
                    self._append_training_metrics(global_step, "decom_warmup", numeric_logs)
                last_iter_end = time.perf_counter()
                accum_data_time = 0.0
                accum_compute_time = 0.0
                accum_samples = 0

        if warmup_steps > 0:
            logger.info("Retinex warmup finished. Transitioning into joint training.")

        completed_joint_steps = max(0, global_step - warmup_steps)
        if global_step >= warmup_steps:
            self._set_training_stage(completed_joint_steps)
        accum_data_time = 0.0
        accum_compute_time = 0.0
        accum_samples = 0
        for epoch in range(first_epoch, self.args.epochs):
            self.training_model.train()
            for batch in train_dataloader:
                if completed_joint_steps >= self.args.max_train_steps:
                    break

                phase = self._set_training_stage(completed_joint_steps)
                data_time = time.perf_counter() - last_iter_end
                compute_start = time.perf_counter()
                _, logs = self._train_step(
                    batch,
                    optimizer,
                    lr_scheduler,
                    global_step=completed_joint_steps,
                    phase=phase,
                    joint_step=completed_joint_steps,
                )
                compute_time = time.perf_counter() - compute_start
                batch_size = batch[0].shape[0]
                accum_data_time += data_time
                accum_compute_time += compute_time
                accum_samples += batch_size

                if self.accelerator.sync_gradients:
                    runtime_metrics = self._collect_runtime_metrics(
                        data_time=accum_data_time,
                        compute_time=accum_compute_time,
                        batch_size=accum_samples,
                    )
                    logs.update(runtime_metrics)
                    logs.update(self.latest_validation_metrics)
                    progress_bar.update(1)
                    global_step += 1
                    completed_joint_steps += 1
                    self._global_step = global_step  # Update for EMA warmup

                    logs["epoch"] = epoch + 1
                    logs["step"] = global_step
                    logs["phase"] = phase
                    numeric_logs = {k: v for k, v in logs.items() if isinstance(v, (int, float))}
                    if self.accelerator.is_main_process and self._should_log_step(global_step, terminal_step=total_training_steps):
                        live_display.update(logs)
                        if not rich_enabled:
                            if phase == "stage1_decom":
                                logger.info(
                                    f"[{phase}] epoch={epoch + 1} step={global_step}/{total_training_steps} "
                                    f"loss={logs['loss']:.4f} l_ret={logs['l_ret']:.4f} "
                                    f"samples/s={logs['samples_per_sec']:.2f} "
                                    f"data={logs['data_time']:.3f}s compute={logs['compute_time']:.3f}s "
                                    f"cpu={logs['cpu_percent']:.1f}% gpu_alloc={logs['gpu_allocated_gb']:.2f}GB "
                                    f"gpu_resv={logs['gpu_reserved_gb']:.2f}GB"
                                )
                            else:
                                logger.info(
                                    f"[{phase}] epoch={epoch + 1} step={global_step}/{total_training_steps} "
                                    f"loss={logs['loss']:.4f} l_diff={logs['l_diff']:.4f} "
                                    f"l_x0={logs['l_x0']:.4f} l_ret={logs['l_ret']:.4f} "
                                    f"samples/s={logs['samples_per_sec']:.2f} "
                                    f"data={logs['data_time']:.3f}s compute={logs['compute_time']:.3f}s "
                                    f"cpu={logs['cpu_percent']:.1f}% gpu_alloc={logs['gpu_allocated_gb']:.2f}GB "
                                    f"gpu_resv={logs['gpu_reserved_gb']:.2f}GB"
                                )
                        self.accelerator.log(numeric_logs, step=global_step)
                        self._append_training_metrics(global_step, phase, numeric_logs)

                    if global_step > 0 and self.args.validation_steps > 0 and global_step % self.args.validation_steps == 0:
                        self.validate(
                            step=global_step,
                            fast=bool(getattr(self.args, "train_fast_validation", True)),
                        )

                    if global_step > 0 and self.args.checkpointing_steps > 0 and global_step % self.args.checkpointing_steps == 0:
                        self._save_checkpoint(global_step)
                    last_iter_end = time.perf_counter()
                    accum_data_time = 0.0
                    accum_compute_time = 0.0
                    accum_samples = 0

            if completed_joint_steps >= self.args.max_train_steps:
                break

        live_display.stop()
        self.accelerator.end_training()
        self._save_final_model(global_step=global_step)

    def _train_step(self, batch, optimizer, lr_scheduler, global_step: int, phase: str, joint_step: int):
        low_light_images, clean_images = batch

        # CFG: Randomly drop conditions during training (10% probability)
        use_cfg = getattr(self.args, 'use_cfg', False)
        if use_cfg and phase not in ["stage1_decom", "decom_warmup"]:
            cfg_drop_prob = getattr(self.args, 'cfg_drop_prob', 0.1)
            drop_mask = torch.rand(low_light_images.shape[0], device=low_light_images.device) < cfg_drop_prob

            # Create unconditional input (zero condition)
            if drop_mask.any():
                low_light_uncond = torch.zeros_like(low_light_images)
                low_light_images = torch.where(
                    drop_mask.view(-1, 1, 1, 1),
                    low_light_uncond,
                    low_light_images
                )

        with self.accelerator.accumulate(self.training_model):
            decom_trainable = phase in {"stage1_decom", "decom_warmup", "stage2_adapter"} or (
                phase == "stage3_joint" and joint_step >= self.args.freeze_decom_steps
            )
            retinex_branch_multiplier = self._retinex_branch_multiplier(phase, joint_step)
            compute_clean_decomposition = phase in {"stage1_decom", "decom_warmup"} or retinex_branch_multiplier > 0.0
            clean_decomposition_requires_grad = decom_trainable and compute_clean_decomposition
            if phase in ["stage1_decom", "decom_warmup"]:
                _, aux = self.training_model(
                    low_light_images,
                    clean_images=clean_images,
                    decomposition_only=True,
                    decomposition_requires_grad=decom_trainable,
                    compute_clean_decomposition=compute_clean_decomposition,
                    clean_decomposition_requires_grad=clean_decomposition_requires_grad,
                )
                retinex_losses = self._compute_retinex_losses(aux)
                loss = retinex_losses["retinex_total"]

                self.accelerator.backward(loss)
                if self.accelerator.sync_gradients:
                    if self._optimizer_has_gradients(optimizer):
                        self.accelerator.clip_grad_norm_(self._params_for_gradient_clipping(), self.args.grad_clip_norm)
                        optimizer.step()
                        lr_scheduler.step()
                        self._ema_step()
                    elif self.accelerator.is_main_process:
                        logger.warning(
                            "Skipping optimizer step in %s at global_step=%s because no gradients were recorded.",
                            phase,
                            global_step,
                        )
                    optimizer.zero_grad()

                logs = {
                    "loss": loss.item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "l_ret": retinex_losses["retinex_total"].item(),
                    "l_recon_low": retinex_losses["l_recon_low"].item(),
                    "l_recon_high": retinex_losses["l_recon_high"].item(),
                    "l_consistency": retinex_losses["l_consistency"].item(),
                    "l_exposure": retinex_losses["l_exposure"].item(),
                    "l_tv": retinex_losses["l_tv"].item(),
                }
                return loss, logs

            residual_target = self._residual_target(clean_images, low_light_images)

            noise = torch.randn_like(residual_target)
            if self.args.offset_noise:
                offset_noise = torch.randn(
                    residual_target.shape[0],
                    residual_target.shape[1],
                    1,
                    1,
                    device=residual_target.device,
                )
                noise = noise + self.args.offset_noise_scale * offset_noise

            bsz = residual_target.shape[0]
            timesteps = torch.randint(
                0,
                self.noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=residual_target.device,
            ).long()
            noisy_target = self.noise_scheduler.add_noise(residual_target, noise, timesteps)

            model_pred, aux = self.training_model(
                low_light_images,
                noisy_target,
                timesteps,
                clean_images=clean_images,
                decomposition_requires_grad=decom_trainable,
                compute_clean_decomposition=compute_clean_decomposition,
                clean_decomposition_requires_grad=clean_decomposition_requires_grad,
            )

            if self.args.prediction_type == "epsilon":
                target = noise
            else:
                target = self.noise_scheduler.get_velocity(residual_target, noise, timesteps)

            weighting_scheme = getattr(self.args, "loss_weighting_scheme", "min_snr")
            if weighting_scheme == "min_snr":
                snr_weights = compute_min_snr_loss_weights(self.noise_scheduler, timesteps, self.args.snr_gamma)
            else:
                snr_weights = compute_adaptive_loss_weights(
                    self.noise_scheduler,
                    timesteps,
                    weighting_scheme=weighting_scheme,
                    snr_gamma=self.args.snr_gamma,
                )
            loss_diff_elem = charbonnier_loss_elementwise(model_pred, target)
            loss_diffusion = (loss_diff_elem.mean(dim=[1, 2, 3]) * snr_weights).mean()

            pred_residual = self._reconstruct_x0(noisy_target, model_pred, timesteps)
            pred_clean = self._decode_residual(low_light_images, pred_residual)

            timestep_mask = timesteps <= self.args.x0_loss_t_max
            x0_loss = torch.tensor(0.0, device=clean_images.device)
            x0_logs = {}
            if torch.any(timestep_mask):
                x0_weights = self._x0_branch_weight(joint_step, timesteps[timestep_mask])
                if isinstance(x0_weights, torch.Tensor) and x0_weights.sum() > 0:
                    pred_clean_masked = pred_clean[timestep_mask]
                    clean_masked = clean_images[timestep_mask]
                    x0_loss, x0_logs = self.criterion(
                        pred_clean_masked,
                        clean_masked,
                        sample_weight=x0_weights,
                    )
                elif isinstance(x0_weights, float) and x0_weights > 0:
                    x0_loss, x0_logs = self.criterion(pred_clean[timestep_mask], clean_images[timestep_mask])
                    x0_loss = x0_loss * x0_weights

            loss = loss_diffusion + x0_loss

            retinex_losses = self._compute_retinex_losses(aux)
            if retinex_branch_multiplier > 0.0:
                loss = loss + retinex_branch_multiplier * retinex_losses["retinex_total"]

            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                if self._optimizer_has_gradients(optimizer):
                    self.accelerator.clip_grad_norm_(self._params_for_gradient_clipping(), self.args.grad_clip_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    self._ema_step()
                elif self.accelerator.is_main_process:
                    logger.warning(
                        "Skipping optimizer step in %s at global_step=%s because no gradients were recorded.",
                        phase,
                        global_step,
                    )
                optimizer.zero_grad()

            # Get x0_weight for logging
            x0_weight_log = self._x0_branch_weight(joint_step)
            if isinstance(x0_weight_log, torch.Tensor):
                x0_weight_log = x0_weight_log.mean().item()

            logs = {
                "loss": loss.item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "l_diff": loss_diffusion.item(),
                "l_x0": x0_loss.item(),
                "l_wavelet": float(x0_logs.get("l_wavelet", torch.tensor(0.0, device=clean_images.device)).item()) if x0_logs else 0.0,
                "x0_w": x0_weight_log,
                "l_ret": retinex_losses["retinex_total"].item(),
                "l_recon_low": retinex_losses["l_recon_low"].item(),
                "l_recon_high": retinex_losses["l_recon_high"].item(),
                "l_consistency": retinex_losses["l_consistency"].item(),
                "l_exposure": retinex_losses["l_exposure"].item(),
                "l_tv": retinex_losses["l_tv"].item(),
            }
            return loss, logs

    def _inference_step(self, low_light, unet, decom, condition_adapter, scheduler, num_inference_steps=None, guidance_scale=None):
        """
        CFG-enabled inference:
        - guidance_scale=1.0: No CFG (conditional only)
        - guidance_scale>1.0: Enhanced conditioning
        """
        step_count = num_inference_steps or self.args.num_inference_steps
        if guidance_scale is None:
            guidance_scale = getattr(self.args, 'guidance_scale', 1.0)

        scheduler.set_timesteps(step_count)
        latents = torch.randn_like(low_light) * scheduler.init_noise_sigma
        inference_model = CombinedModel(
            unet,
            decom_model=decom,
            condition_adapter=condition_adapter,
            conditioning_space=self.args.conditioning_space,
            inject_mode=getattr(self.args, "inject_mode", "concat_pyramid"),
        )

        use_cfg = guidance_scale > 1.0

        for timestep in scheduler.timesteps:
            latent_input = scheduler.scale_model_input(latents, timestep)

            if use_cfg:
                # Conditional prediction
                model_input_cond, aux_cond = inference_model.build_model_input(low_light, latent_input)
                noise_pred_cond = inference_model.run_unet(model_input_cond, timestep, aux_cond)

                # Unconditional prediction (zero condition)
                low_light_uncond = torch.zeros_like(low_light)
                model_input_uncond, aux_uncond = inference_model.build_model_input(low_light_uncond, latent_input)
                noise_pred_uncond = inference_model.run_unet(model_input_uncond, timestep, aux_uncond)

                # CFG formula
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                model_input, aux = inference_model.build_model_input(low_light, latent_input)
                noise_pred = inference_model.run_unet(model_input, timestep, aux)

            latents = scheduler.step(noise_pred, timestep, latents).prev_sample

        enhanced = self._decode_residual(low_light, latents)
        return (enhanced / 2 + 0.5).clamp(0, 1)

    def _save_validation_grid(self, step, step_count, low_01, enhanced_01, clean_01):
        if not self.accelerator.is_main_process:
            return

        val_dir = os.path.join(self.args.output_dir, "validation")
        os.makedirs(val_dir, exist_ok=True)

        grid = torch.cat([low_01, enhanced_01, clean_01], dim=0)
        grid_image = make_grid(grid, nrow=low_01.shape[0], padding=2)
        grid_path = os.path.join(val_dir, f"val_step_{step}_steps_{step_count}.png")
        transforms.ToPILImage()(grid_image).save(grid_path)
        logger.info(f"Saved validation grid to {grid_path}")

    def _validate_for_step_count(
        self,
        eval_dataloader,
        unet,
        decom,
        condition_adapter,
        step_count: int,
        step=None,
        lpips_fn=None,
        semantic_metric=None,
        compute_niqe: bool = True,
    ):
        scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config)

        total_psnr = 0.0
        total_ssim = 0.0
        total_lpips = 0.0
        total_semantic = 0.0
        total_niqe = 0.0
        niqe_count = 0
        total_elapsed = 0.0
        num_samples = 0
        saved_grid = False

        with torch.no_grad():
            for val_low, val_clean in eval_dataloader:
                if self.accelerator.is_main_process and num_samples >= self.args.num_validation_images:
                    break

                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.accelerator.device)
                start_time = time.perf_counter()
                enhanced = self._inference_step(val_low, unet, decom, condition_adapter, scheduler, num_inference_steps=step_count)
                if torch.cuda.is_available():
                    torch.cuda.synchronize(self.accelerator.device)
                elapsed = time.perf_counter() - start_time

                elapsed_tensor = torch.tensor([elapsed], device=val_low.device)
                gathered_elapsed = self.accelerator.gather_for_metrics(elapsed_tensor)
                gathered_low = self.accelerator.gather_for_metrics(val_low)
                gathered_clean = self.accelerator.gather_for_metrics(val_clean)
                gathered_enhanced = self.accelerator.gather_for_metrics(enhanced)

                if not self.accelerator.is_main_process:
                    continue

                remaining = self.args.num_validation_images - num_samples
                if remaining <= 0:
                    break

                gathered_low = gathered_low[:remaining]
                gathered_clean = gathered_clean[:remaining]
                gathered_enhanced = gathered_enhanced[:remaining]

                low_01 = (gathered_low / 2 + 0.5).clamp(0, 1)
                clean_01 = (gathered_clean / 2 + 0.5).clamp(0, 1)
                enhanced_01 = gathered_enhanced

                batch_size = enhanced_01.shape[0]
                total_psnr += peak_signal_noise_ratio(enhanced_01, clean_01, data_range=1.0).item() * batch_size
                total_ssim += ssim(enhanced_01, clean_01).item() * batch_size

                if lpips_fn is not None:
                    enhanced_11 = enhanced_01 * 2 - 1
                    clean_11 = clean_01 * 2 - 1
                    total_lpips += lpips_fn(enhanced_11, clean_11).mean().item() * batch_size

                if semantic_metric is not None and semantic_metric.available:
                    semantic_distance = semantic_metric.compute(enhanced_01.to(self.accelerator.device), clean_01.to(self.accelerator.device))
                    if semantic_distance is not None:
                        total_semantic += semantic_distance * batch_size

                if compute_niqe and self.args.nr_metric == "niqe":
                    niqe_score = try_compute_niqe(enhanced_01)
                    if niqe_score is not None:
                        total_niqe += niqe_score * batch_size
                        niqe_count += batch_size

                total_elapsed += float(gathered_elapsed.max().item())
                num_samples += batch_size

                if not saved_grid and step_count == self.args.num_inference_steps:
                    self._save_validation_grid(step, step_count, low_01, enhanced_01, clean_01)
                    saved_grid = True

        if not self.accelerator.is_main_process or num_samples == 0:
            return None

        metrics = {
            "psnr": total_psnr / num_samples,
            "ssim": total_ssim / num_samples,
            "lpips": total_lpips / num_samples if lpips_fn is not None else None,
            "semantic_distance": total_semantic / num_samples if semantic_metric is not None and semantic_metric.available else None,
            "niqe": total_niqe / niqe_count if niqe_count > 0 else None,
            "seconds_per_image": total_elapsed / num_samples,
        }
        return metrics

    def validate(self, step=None, fast: bool = False):
        mode_label = "fast" if fast else "full"
        logger.info(f"Running {mode_label} validation at step {step}...")

        eval_dataloader = self._get_eval_dataloader()
        unwrapped = self._unwrap_training_model()
        unet = unwrapped.unet
        decom = unwrapped.decom_model
        condition_adapter = unwrapped.condition_adapter

        self._ema_store()
        self._ema_copy_to_modules()

        unet.eval()
        if decom is not None:
            decom.eval()
        if condition_adapter is not None:
            condition_adapter.eval()

        benchmark_steps, metric_names = self._validation_plan(fast)
        compute_lpips = "lpips" in metric_names
        compute_semantic = "semantic_distance" in metric_names
        compute_niqe = "niqe" in metric_names and self.args.nr_metric == "niqe"

        lpips_fn = self._get_lpips_fn() if compute_lpips else None
        semantic_metric = self._get_semantic_metric() if compute_semantic else None

        validation_results = {}
        validation_started_at = time.perf_counter()
        if self.accelerator.is_main_process:
            logger.info(
                "Validation plan: steps=%s metrics=%s",
                benchmark_steps,
                list(metric_names),
            )

        for step_count in benchmark_steps:
            result = self._validate_for_step_count(
                eval_dataloader,
                unet,
                decom,
                condition_adapter,
                step_count=step_count,
                step=step,
                lpips_fn=lpips_fn,
                semantic_metric=semantic_metric,
                compute_niqe=compute_niqe,
            )
            if result is not None:
                validation_results[step_count] = result

        if self.accelerator.is_main_process and validation_results:
            primary_metrics = validation_results.get(self.args.num_inference_steps, next(iter(validation_results.values())))
            self.latest_validation_metrics = {
                "val_psnr": primary_metrics.get("psnr"),
                "val_ssim": primary_metrics.get("ssim"),
                "val_lpips": primary_metrics.get("lpips"),
                "val_step": step,
            }
            primary_seconds_per_image = primary_metrics.get("seconds_per_image")

            log_dict = {}
            lines = []
            for step_count, metrics in validation_results.items():
                lines.append(f"[steps={step_count}]")
                lines.append(f"PSNR: {metrics['psnr']:.4f}")
                lines.append(f"SSIM: {metrics['ssim']:.4f}")
                lines.append(f"SecondsPerImage: {metrics['seconds_per_image']:.6f}")

                log_dict[f"val/psnr_{step_count}step"] = metrics["psnr"]
                log_dict[f"val/ssim_{step_count}step"] = metrics["ssim"]
                log_dict[f"val/sec_per_image_{step_count}step"] = metrics["seconds_per_image"]

                if metrics["lpips"] is not None:
                    lines.append(f"LPIPS: {metrics['lpips']:.4f}")
                    log_dict[f"val/lpips_{step_count}step"] = metrics["lpips"]
                if metrics["semantic_distance"] is not None:
                    lines.append(f"SemanticDistance: {metrics['semantic_distance']:.4f}")
                    log_dict[f"val/semantic_distance_{step_count}step"] = metrics["semantic_distance"]
                if metrics["niqe"] is not None:
                    lines.append(f"NIQE: {metrics['niqe']:.4f}")
                    log_dict[f"val/niqe_{step_count}step"] = metrics["niqe"]
                lines.append("")

            self.accelerator.log(log_dict, step=step or 0)
            self._write_status_snapshot({
                "step": step or 0,
                "phase": "validation",
                **self.latest_validation_metrics,
            })
            self._append_training_metrics(
                step or 0,
                "validation",
                {
                    "epoch": "",
                    "loss": "",
                    "lr": "",
                    "l_diff": "",
                    "l_x0": "",
                    "l_wavelet": "",
                    "x0_w": "",
                    "l_ret": "",
                    "l_recon_low": "",
                    "l_recon_high": "",
                    "l_consistency": "",
                    "l_exposure": "",
                    "l_tv": "",
                    "data_time": "",
                    "compute_time": time.perf_counter() - validation_started_at,
                    "iter_time": "",
                    "data_wait_ratio": "",
                    "samples_per_sec": "",
                    "cpu_percent": "",
                    "cpu_rss_gb": "",
                    "gpu_allocated_gb": "",
                    "gpu_reserved_gb": "",
                    "gpu_max_reserved_gb": "",
                    "val_psnr": primary_metrics.get("psnr"),
                    "val_ssim": primary_metrics.get("ssim"),
                    "val_lpips": primary_metrics.get("lpips"),
                    "val_seconds_per_image": primary_seconds_per_image,
                    "val_step_count": self.args.num_inference_steps,
                },
            )

            metrics_path = os.path.join(self.args.output_dir, "metrics.txt")
            with open(metrics_path, "w") as metrics_file:
                metrics_file.write("\n".join(lines).strip() + "\n")
                metrics_file.write(f"Step: {step}\n")

            if primary_metrics["psnr"] > self.best_psnr:
                self.best_psnr = primary_metrics["psnr"]
                logger.info(f"New best model found (PSNR: {primary_metrics['psnr']:.4f}). Saving...")
                self._save_model_bundle(
                    os.path.join(self.args.output_dir, "best_model"),
                    unet,
                    decom,
                    condition_adapter,
                    prefix="best",
                    step=step,
                    metrics=primary_metrics,
                )

        self._ema_restore()

        self.training_model.train()
        self.accelerator.wait_for_everyone()
        return validation_results

    def predict(self):
        self.unet.eval()
        if self.decom_model is not None:
            self.decom_model.eval()
        if self.condition_adapter is not None:
            self.condition_adapter.eval()

        scheduler = DPMSolverMultistepScheduler.from_config(self.noise_scheduler.config)

        if self.args.video_path:
            self._predict_video(scheduler)
        else:
            self._predict_image(scheduler)

    def _predict_image(self, scheduler):
        logger.info(f"Starting image prediction from {self.args.data_dir}")
        dataset = LowLightDataset(image_dir=self.args.data_dir, img_size=self.args.resolution, phase="predict")
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=False, num_workers=self.args.num_workers)

        out_dir = os.path.join(self.args.output_dir, "predictions")
        os.makedirs(out_dir, exist_ok=True)

        to_pil = transforms.ToPILImage()
        index = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting"):
                low_light = batch[0] if isinstance(batch, (list, tuple)) else batch
                low_light = low_light.to(self.accelerator.device)
                enhanced = self._inference_step(
                    low_light,
                    self.unet,
                    self.decom_model,
                    self.condition_adapter,
                    scheduler,
                    num_inference_steps=self.args.num_inference_steps,
                )

                for item in enhanced:
                    to_pil(item.cpu()).save(os.path.join(out_dir, f"enhanced_{index:05d}.png"))
                    index += 1

        logger.info(f"Saved prediction results to {out_dir}")

    def _predict_video(self, scheduler):
        logger.info(f"Starting video prediction: {self.args.video_path}")
        if not os.path.exists(self.args.video_path):
            raise FileNotFoundError(f"Video not found: {self.args.video_path}")

        capture = cv2.VideoCapture(self.args.video_path)
        fps = capture.get(cv2.CAP_PROP_FPS)

        frames_dir = os.path.join(self.args.output_dir, "video_frames")
        os.makedirs(frames_dir, exist_ok=True)

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((self.args.resolution, self.args.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

        frame_index = 0
        with torch.no_grad():
            while True:
                success, frame = capture.read()
                if not success:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                input_tensor = transform(frame_rgb).unsqueeze(0).to(self.accelerator.device)
                enhanced = self._inference_step(
                    input_tensor,
                    self.unet,
                    self.decom_model,
                    self.condition_adapter,
                    scheduler,
                    num_inference_steps=self.args.num_inference_steps,
                )

                enhanced_np = enhanced.squeeze(0).cpu().numpy().transpose(1, 2, 0)
                enhanced_bgr = cv2.cvtColor((enhanced_np * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_index:06d}.png"), enhanced_bgr)
                frame_index += 1

        capture.release()

        out_video = os.path.join(self.args.output_dir, "enhanced.mp4")
        video_writer(frames_dir, out_video, fps=fps)
        logger.info(f"Video saved to {out_video}")

    def _model_metadata(self, step=None, metrics=None):
        serializable_args = {}
        for key, value in vars(self.args).items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                serializable_args[key] = value
            elif isinstance(value, list):
                serializable_args[key] = value
            else:
                serializable_args[key] = str(value)

        return {
            "step": step,
            "ema_enabled": self.args.use_ema,
            "best_psnr": self.best_psnr,
            "metrics": metrics or {},
            "effective_batch_size": getattr(self.args, "effective_batch_size", None),
            "unet_backend": self.unet_runtime_backend.as_dict() if self.unet_runtime_backend is not None else None,
            "args": serializable_args,
        }

    def _save_model_bundle(self, base_dir, unet, decom, condition_adapter, prefix: str, step=None, metrics=None):
        if not self.accelerator.is_main_process:
            return

        os.makedirs(base_dir, exist_ok=True)
        unet = self._unwrap_compiled_module(unet)
        unet_dirname = f"unet_{prefix}"
        decom_filename = f"decom_model_{prefix}.pth"
        adapter_filename = f"condition_adapter_{prefix}.pth"
        metadata_filename = f"{prefix}_metadata.json"

        if prefix == "final":
            unet_dirname = "unet_final"
            decom_filename = "decom_model.pth"
            adapter_filename = "condition_adapter.pth"
            metadata_filename = "metadata.json"

        unet.save_pretrained(os.path.join(base_dir, unet_dirname))

        if decom is not None:
            torch.save(decom.state_dict(), os.path.join(base_dir, decom_filename))
        if condition_adapter is not None:
            torch.save(condition_adapter.state_dict(), os.path.join(base_dir, adapter_filename))

        metadata_path = os.path.join(base_dir, metadata_filename)
        with open(metadata_path, "w") as metadata_file:
            json.dump(self._model_metadata(step=step, metrics=metrics), metadata_file, indent=2)

    def _save_checkpoint(self, step):
        save_path = os.path.join(self.args.output_dir, f"checkpoint-{step}")
        self.accelerator.save_state(save_path)

        self._save_ema_state(save_path)

        unwrapped = self._unwrap_training_model()
        if unwrapped.decom_model is not None:
            torch.save(unwrapped.decom_model.state_dict(), os.path.join(save_path, "decom_model.pth"))
        if unwrapped.condition_adapter is not None:
            torch.save(unwrapped.condition_adapter.state_dict(), os.path.join(save_path, "condition_adapter.pth"))

        with open(os.path.join(save_path, "checkpoint_metadata.json"), "w") as metadata_file:
            json.dump(self._model_metadata(step=step), metadata_file, indent=2)

        total_limit = getattr(self.args, "checkpoints_total_limit", None)
        if total_limit is not None and self.accelerator.is_main_process:
            checkpoints = sorted(
                [item for item in os.listdir(self.args.output_dir) if item.startswith("checkpoint-")],
                key=lambda item: int(item.split("-")[1]),
            )
            if len(checkpoints) > total_limit:
                for checkpoint_name in checkpoints[: len(checkpoints) - total_limit]:
                    checkpoint_path = os.path.join(self.args.output_dir, checkpoint_name)
                    logger.info(f"Removing old checkpoint: {checkpoint_path}")
                    shutil.rmtree(checkpoint_path)

    def _save_final_model(self, global_step=None):
        if not self.accelerator.is_main_process:
            return

        self._ema_store()
        self._ema_copy_to_modules()
        unwrapped = self._unwrap_training_model()
        try:
            self._save_model_bundle(
                self.args.output_dir,
                self._unwrap_compiled_module(unwrapped.unet),
                unwrapped.decom_model,
                unwrapped.condition_adapter,
                prefix="final",
                step=global_step,
            )
        finally:
            self._ema_restore()
