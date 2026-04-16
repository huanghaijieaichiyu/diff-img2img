import torch
import torch.nn as nn
import torch.nn.functional as F

from .conditioning import rgb_to_hvi_lite
from utils.runtime_backend import has_compiled_module


class UNetBlockFiLMContext:
    def __init__(self, unet: nn.Module, film_state: dict | None):
        self.unet = unet
        self.film_state = film_state or {}
        self.handles = []

    @staticmethod
    def _apply_film(tensor: torch.Tensor, gamma_beta: torch.Tensor | None):
        if gamma_beta is None or not torch.is_tensor(tensor):
            return tensor
        if gamma_beta.shape[-2:] != tensor.shape[-2:]:
            gamma_beta = F.interpolate(gamma_beta, size=tensor.shape[-2:], mode="bilinear", align_corners=False)
        if gamma_beta.shape[1] != tensor.shape[1] * 2:
            return tensor
        gamma, beta = gamma_beta.chunk(2, dim=1)
        return tensor * (1.0 + torch.tanh(gamma)) + beta

    def _patch_output(self, output, gamma_beta):
        if gamma_beta is None:
            return output
        if torch.is_tensor(output):
            return self._apply_film(output, gamma_beta)
        if isinstance(output, tuple):
            patched = list(output)
            if patched and torch.is_tensor(patched[0]):
                patched[0] = self._apply_film(patched[0], gamma_beta)
            return tuple(patched)
        return output

    def _make_hook(self, gamma_beta):
        def hook(_module, _inputs, output):
            return self._patch_output(output, gamma_beta)
        return hook

    def __enter__(self):
        for index, block in enumerate(self.unet.down_blocks):
            gamma_beta = self.film_state.get("down", [None] * len(self.unet.down_blocks))[index]
            self.handles.append(block.register_forward_hook(self._make_hook(gamma_beta)))

        if hasattr(self.unet, "mid_block") and self.unet.mid_block is not None:
            self.handles.append(self.unet.mid_block.register_forward_hook(self._make_hook(self.film_state.get("mid"))))

        for index, block in enumerate(self.unet.up_blocks):
            gamma_beta = self.film_state.get("up", [None] * len(self.unet.up_blocks))[index]
            self.handles.append(block.register_forward_hook(self._make_hook(gamma_beta)))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class CombinedModel(nn.Module):
    def __init__(
        self,
        unet,
        decom_model=None,
        condition_adapter=None,
        conditioning_space: str = "hvi_lite",
        inject_mode: str = "concat_pyramid",
    ):
        super().__init__()
        self.unet = unet
        self.decom_model = decom_model
        self.condition_adapter = condition_adapter
        self.conditioning_space = conditioning_space
        self.inject_mode = inject_mode

    def _uses_compiled_unet(self) -> bool:
        return has_compiled_module(self.unet)

    @staticmethod
    def _mark_cudagraph_step_begin():
        compiler_ns = getattr(torch, "compiler", None)
        mark_step = getattr(compiler_ns, "cudagraph_mark_step_begin", None)
        if callable(mark_step):
            mark_step()

    def _build_condition_space(self, low_light_01: torch.Tensor) -> torch.Tensor:
        if self.condition_adapter is not None and hasattr(self.condition_adapter, "build_condition_space"):
            return self.condition_adapter.build_condition_space(low_light_01, conditioning_space=self.conditioning_space)
        if self.conditioning_space == "hvi_lite":
            return rgb_to_hvi_lite(low_light_01)
        return low_light_01

    def _decompose(self, images_01: torch.Tensor, *, requires_grad: bool = True):
        if self.decom_model is None:
            return None, None
        if requires_grad:
            return self.decom_model(images_01)
        with torch.no_grad():
            return self.decom_model(images_01)

    def _build_auxiliary(
        self,
        low_light_images: torch.Tensor,
        clean_images: torch.Tensor | None = None,
        *,
        decomposition_requires_grad: bool = True,
        compute_clean_decomposition: bool = True,
        clean_decomposition_requires_grad: bool = True,
    ):
        low_light_01 = (low_light_images / 2 + 0.5).clamp(0, 1)
        aux = {
            "low_light_01": low_light_01,
            "conditioning_space": self.conditioning_space,
        }

        r_low, i_low = self._decompose(low_light_01, requires_grad=decomposition_requires_grad)
        aux["r_low"] = r_low
        aux["i_low"] = i_low

        if clean_images is not None and compute_clean_decomposition:
            clean_images_01 = (clean_images / 2 + 0.5).clamp(0, 1)
            aux["clean_images_01"] = clean_images_01
            r_high, i_high = self._decompose(clean_images_01, requires_grad=clean_decomposition_requires_grad)
            aux["r_high"] = r_high
            aux["i_high"] = i_high

        cond_space = self._build_condition_space(low_light_01)
        aux["low_condition_space"] = cond_space

        cond_intensity = cond_space[:, :1]
        cond_chroma = cond_space[:, 1:]

        if self.decom_model is not None and r_low is not None and i_low is not None:
            illumination_input = torch.cat([i_low, cond_intensity], dim=1)
            detail_input = torch.cat([r_low, cond_chroma], dim=1)
        else:
            illumination_input = cond_intensity
            detail_input = torch.cat([low_light_01, cond_chroma], dim=1)

        aux["illumination_input"] = illumination_input
        aux["detail_input"] = detail_input
        return aux

    def build_model_input(
        self,
        low_light_images: torch.Tensor,
        noisy_images: torch.Tensor,
        clean_images: torch.Tensor | None = None,
        *,
        decomposition_requires_grad: bool = True,
        compute_clean_decomposition: bool = True,
        clean_decomposition_requires_grad: bool = True,
    ):
        aux = self._build_auxiliary(
            low_light_images,
            clean_images=clean_images,
            decomposition_requires_grad=decomposition_requires_grad,
            compute_clean_decomposition=compute_clean_decomposition,
            clean_decomposition_requires_grad=clean_decomposition_requires_grad,
        )

        if self.condition_adapter is not None:
            adapter_output = self.condition_adapter(noisy_images, aux["illumination_input"], aux["detail_input"])
            model_input = torch.cat([noisy_images + adapter_output["noisy_delta"], adapter_output["condition_map"]], dim=1)
            aux["noisy_delta"] = adapter_output["noisy_delta"]
            aux["condition_map"] = adapter_output["condition_map"]
            aux["condition_pyramid"] = adapter_output["condition_pyramid"]
            aux["film_state"] = adapter_output["film_state"]
        elif self.decom_model is not None and aux["r_low"] is not None and aux["i_low"] is not None:
            r_low_norm = aux["r_low"] * 2.0 - 1.0
            i_low_norm = aux["i_low"] * 2.0 - 1.0
            model_input = torch.cat([noisy_images, r_low_norm, i_low_norm], dim=1)
        else:
            model_input = torch.cat([noisy_images, low_light_images], dim=1)

        return model_input, aux

    def run_unet(self, model_input: torch.Tensor, timesteps: torch.Tensor, aux: dict):
        compiled_unet = self._uses_compiled_unet()
        if compiled_unet:
            # `torch.compile(mode="reduce-overhead")` may reuse CUDAGraph outputs
            # across invocations unless we mark explicit step boundaries.
            self._mark_cudagraph_step_begin()

        film_state = aux.get("film_state")
        if self.inject_mode == "film_pyramid" and film_state:
            with UNetBlockFiLMContext(self.unet, film_state):
                sample = self.unet(model_input, timesteps).sample
        else:
            sample = self.unet(model_input, timesteps).sample

        # Clone compiled outputs outside the graph so later invocations do not
        # overwrite tensors that autograd still needs for backward.
        if compiled_unet:
            sample = sample.clone()
        return sample

    def forward(
        self,
        low_light_images: torch.Tensor,
        noisy_images: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        clean_images: torch.Tensor | None = None,
        decomposition_only: bool = False,
        decomposition_requires_grad: bool = True,
        compute_clean_decomposition: bool = True,
        clean_decomposition_requires_grad: bool = True,
    ):
        if decomposition_only:
            return None, self._build_auxiliary(
                low_light_images,
                clean_images=clean_images,
                decomposition_requires_grad=decomposition_requires_grad,
                compute_clean_decomposition=compute_clean_decomposition,
                clean_decomposition_requires_grad=clean_decomposition_requires_grad,
            )

        if noisy_images is None or timesteps is None:
            raise ValueError("noisy_images and timesteps are required for diffusion forward.")

        model_input, aux = self.build_model_input(
            low_light_images,
            noisy_images,
            clean_images=clean_images,
            decomposition_requires_grad=decomposition_requires_grad,
            compute_clean_decomposition=compute_clean_decomposition,
            clean_decomposition_requires_grad=clean_decomposition_requires_grad,
        )
        return self.run_unet(model_input, timesteps, aux), aux
