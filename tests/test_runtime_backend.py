from __future__ import annotations

from types import SimpleNamespace

import pytest

from utils.runtime_backend import has_compiled_module, resolve_unet_runtime_backend


def test_auto_backend_uses_sdpa_for_dynamic_film_hooks_when_compile_is_blocked():
    backend = resolve_unet_runtime_backend(
        SimpleNamespace(
            attention_backend="auto",
            use_torch_compile=True,
            torch_compile_mode="max-autotune-no-cudagraphs",
            enable_torch_sdpa_memory_efficient_attention=True,
            inject_mode="film_pyramid",
            allow_unsafe_compile_with_film=False,
        )
    )

    assert backend.resolved_backend == "sdpa"
    assert backend.compile_enabled is False
    assert backend.sdpa_enabled is True
    assert any("film_pyramid" in reason for reason in backend.reasons)


def test_auto_backend_uses_compile_when_hooks_are_not_dynamic():
    backend = resolve_unet_runtime_backend(
        SimpleNamespace(
            attention_backend="auto",
            use_torch_compile=True,
            torch_compile_mode="max-autotune-no-cudagraphs",
            enable_torch_sdpa_memory_efficient_attention=False,
            inject_mode="concat_pyramid",
            allow_unsafe_compile_with_film=False,
        )
    )

    assert backend.resolved_backend == "compile"
    assert backend.compile_enabled is True
    assert backend.sdpa_enabled is False


def test_explicit_sdpa_backend_enables_torch_attention_processors():
    backend = resolve_unet_runtime_backend(
        SimpleNamespace(
            attention_backend="sdpa",
            use_torch_compile=False,
            torch_compile_mode="max-autotune-no-cudagraphs",
            enable_torch_sdpa_memory_efficient_attention=False,
            inject_mode="concat_pyramid",
            allow_unsafe_compile_with_film=False,
        )
    )

    assert backend.resolved_backend == "sdpa"
    assert backend.compile_enabled is False
    assert backend.sdpa_enabled is True


def test_explicit_compile_backend_rejects_dynamic_film_hooks():
    with pytest.raises(ValueError, match="film_pyramid"):
        resolve_unet_runtime_backend(
            SimpleNamespace(
                attention_backend="compile",
                use_torch_compile=True,
                torch_compile_mode="max-autotune-no-cudagraphs",
                enable_torch_sdpa_memory_efficient_attention=True,
                inject_mode="film_pyramid",
                allow_unsafe_compile_with_film=False,
            )
        )


def test_has_compiled_module_detects_in_place_module_compile():
    import torch.nn as nn

    module = nn.Linear(4, 4)
    assert has_compiled_module(module) is False

    module.compile(mode="max-autotune-no-cudagraphs")
    assert has_compiled_module(module) is True
