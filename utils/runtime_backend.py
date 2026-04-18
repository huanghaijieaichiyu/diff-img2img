from __future__ import annotations

from dataclasses import dataclass, replace

import torch


UNET_BACKEND_CHOICES = ("auto", "compile", "sdpa", "native")
TORCH_COMPILE_MODE_CHOICES = (
    "default",
    "max-autotune",
    "max-autotune-no-cudagraphs",
    "reduce-overhead",
)


@dataclass(frozen=True)
class UNetRuntimeBackend:
    requested_backend: str
    resolved_backend: str
    use_torch_compile: bool
    torch_compile_mode: str
    compile_enabled: bool
    sdpa_requested: bool
    sdpa_enabled: bool
    uses_dynamic_film_hooks: bool
    allow_unsafe_compile_with_film: bool
    reasons: tuple[str, ...] = ()

    def with_reason(self, message: str) -> "UNetRuntimeBackend":
        return replace(self, reasons=self.reasons + (message,))

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_backend": self.requested_backend,
            "resolved_backend": self.resolved_backend,
            "use_torch_compile": self.use_torch_compile,
            "torch_compile_mode": self.torch_compile_mode,
            "compile_enabled": self.compile_enabled,
            "sdpa_requested": self.sdpa_requested,
            "sdpa_enabled": self.sdpa_enabled,
            "uses_dynamic_film_hooks": self.uses_dynamic_film_hooks,
            "allow_unsafe_compile_with_film": self.allow_unsafe_compile_with_film,
            "reasons": list(self.reasons),
        }


def has_compiled_module(module) -> bool:
    if module is None:
        return False
    if hasattr(module, "_orig_mod"):
        return True
    compiled_call = getattr(module, "_compiled_call_impl", None)
    return compiled_call is not None


def unwrap_compiled_module(module):
    if hasattr(module, "_orig_mod"):
        return module._orig_mod
    return module


def _requested_backend(args) -> str:
    backend = getattr(args, "attention_backend", None) or "auto"
    backend = str(backend).strip().lower()
    if backend not in UNET_BACKEND_CHOICES:
        raise ValueError(
            f"Unsupported attention_backend={backend!r}. "
            f"Expected one of {UNET_BACKEND_CHOICES}."
        )
    return backend


def _torch_compile_mode(args) -> str:
    mode = getattr(args, "torch_compile_mode",
                   None) or "max-autotune-no-cudagraphs"
    mode = str(mode).strip().lower()
    if mode not in TORCH_COMPILE_MODE_CHOICES:
        raise ValueError(
            f"Unsupported torch_compile_mode={mode!r}. "
            f"Expected one of {TORCH_COMPILE_MODE_CHOICES}."
        )
    return mode


def resolve_unet_runtime_backend(args) -> UNetRuntimeBackend:
    requested_backend = _requested_backend(args)
    use_torch_compile = bool(getattr(args, "use_torch_compile", True))
    torch_compile_mode = _torch_compile_mode(args)
    sdpa_requested = bool(
        getattr(args, "enable_torch_sdpa_memory_efficient_attention", False))
    uses_dynamic_film_hooks = getattr(
        args, "inject_mode", None) == "film_pyramid"
    allow_unsafe_compile_with_film = bool(
        getattr(args, "allow_unsafe_compile_with_film", False))
    compile_supported = hasattr(torch, "compile") and callable(
        getattr(torch.nn.Module, "compile", None))

    backend = UNetRuntimeBackend(
        requested_backend=requested_backend,
        resolved_backend="native",
        use_torch_compile=use_torch_compile,
        torch_compile_mode=torch_compile_mode,
        compile_enabled=False,
        sdpa_requested=sdpa_requested,
        sdpa_enabled=False,
        uses_dynamic_film_hooks=uses_dynamic_film_hooks,
        allow_unsafe_compile_with_film=allow_unsafe_compile_with_film,
    )

    compile_requested = requested_backend == "compile" or (
        requested_backend == "auto" and use_torch_compile
    )
    compile_block_reason = None

    if compile_requested and not compile_supported:
        compile_block_reason = "torch.compile is unavailable in this PyTorch build."
    elif compile_requested and uses_dynamic_film_hooks and not allow_unsafe_compile_with_film:
        compile_block_reason = (
            "inject_mode=film_pyramid relies on per-step FiLM hook state, "
            "which is unsafe to combine with torch.compile by default."
        )

    if requested_backend == "compile":
        if compile_block_reason is not None:
            raise ValueError(
                "attention_backend=compile cannot be used in the current configuration: "
                f"{compile_block_reason}"
            )
        return replace(
            backend,
            resolved_backend="compile",
            compile_enabled=True,
            sdpa_enabled=sdpa_requested,
        ).with_reason("Explicit compile backend requested.")

    if requested_backend == "sdpa":
        return replace(
            backend,
            resolved_backend="sdpa",
            sdpa_enabled=True,
        ).with_reason("Explicit PyTorch SDPA backend requested.")

    if requested_backend == "native":
        return backend.with_reason("Explicit native PyTorch attention backend requested.")

    if compile_requested and compile_block_reason is None:
        return replace(
            backend,
            resolved_backend="compile",
            compile_enabled=True,
            sdpa_enabled=sdpa_requested,
        ).with_reason("Auto backend selected torch.compile.")

    if compile_block_reason is not None:
        backend = backend.with_reason(compile_block_reason)

    if sdpa_requested:
        return replace(
            backend,
            resolved_backend="sdpa",
            sdpa_enabled=True,
        ).with_reason("Falling back to PyTorch SDPA attention.")

    return backend.with_reason("Falling back to native PyTorch attention.")
