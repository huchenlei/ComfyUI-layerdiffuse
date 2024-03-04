import os
from enum import Enum
import torch
import functools

import folder_paths
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from comfy.conds import CONDRegular
from .lib_layerdiffusion.utils import (
    rgba2rgbfp32,
    load_file_from_url,
    to_lora_patch_dict,
)
from .lib_layerdiffusion.models import TransparentVAEDecoder


layer_model_root = os.path.join(folder_paths.models_dir, "layer_model")
load_layer_model_state_dict = load_torch_file


def calculate_weight_adjust_channel(func):
    @functools.wraps(func)
    def calculate_weight(
        self: ModelPatcher, patches, weight: torch.Tensor, key: str
    ) -> torch.Tensor:
        weight = func(self, patches, weight, key)

        for p in patches:
            alpha = p[0]
            v = p[1]

            if isinstance(v, list):
                v = (func(v[1:], v[0].clone(), key),)

            if len(v) == 1:
                patch_type = "diff"
            elif len(v) == 2:
                patch_type = v[0]
                v = v[1]

            if patch_type == "diff":
                w1 = v[0]
                if all(
                    (
                        alpha != 0.0,
                        w1.shape != weight.shape,
                        w1.ndim == weight.ndim == 4,
                    )
                ):
                    new_shape = [max(n, m) for n, m in zip(weight.shape, w1.shape)]
                    print(f"Merged with {key} channel changed from {weight.shape} to {new_shape}")
                    new_diff = alpha * comfy.model_management.cast_to_device(
                        w1, weight.device, weight.dtype
                    )
                    new_weight = torch.zeros(size=new_shape).to(weight)
                    new_weight[
                        : weight.shape[0],
                        : weight.shape[1],
                        : weight.shape[2],
                        : weight.shape[3],
                    ] = weight
                    new_weight[
                        : new_diff.shape[0],
                        : new_diff.shape[1],
                        : new_diff.shape[2],
                        : new_diff.shape[3],
                    ] += new_diff
                    new_weight = new_weight.contiguous().clone()
                    weight = new_weight
        return weight

    return calculate_weight


ModelPatcher.calculate_weight = calculate_weight_adjust_channel(
    ModelPatcher.calculate_weight
)


class RGBA2RBGfp32:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "rgb2rgb_fp32"
    CATEGORY = "layered_diffusion"

    def rgb2rgb_fp32(self, image):
        return rgba2rgbfp32(image)


class LayeredDiffusionDecode:
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    Outputs RGB image + Alpha mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "decode"
    CATEGORY = "layered_diffusion"

    def __init__(self) -> None:
        self.vae_transparent_decoder = None

    def decode(self, samples, images):
        if self.vae_transparent_decoder is None:
            model_path = load_file_from_url(
                url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors",
                model_dir=layer_model_root,
                file_name="vae_transparent_decoder.safetensors",
            )
            self.vae_transparent_decoder = TransparentVAEDecoder(
                load_torch_file(model_path),
                device=comfy.model_management.get_torch_device(),
                dtype=(
                    torch.float16
                    if comfy.model_management.should_use_fp16()
                    else torch.float32
                ),
            )
        latent = samples["samples"]
        pixel = images.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]
        pixel_with_alpha = self.vae_transparent_decoder.decode_pixel(pixel, latent)
        # [B, C, H, W] => [B, H, W, C]
        pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
        image = pixel_with_alpha[..., 1:]
        alpha = pixel_with_alpha[..., 0]
        return (image, alpha)


class LayeredDiffusionDecodeRGBA(LayeredDiffusionDecode):
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    Outputs RGBA image.
    """

    RETURN_TYPES = ("IMAGE",)

    def decode(self, samples, images):
        image, mask = super().decode(samples, images)
        alpha = 1.0 - mask
        return JoinImageWithAlpha().join_image_with_alpha(image, alpha)


class LayerMethod(Enum):
    ATTN = "Attention Injection"
    CONV = "Conv Injection"


class LayerType(Enum):
    FG = "Foreground"
    BG = "Background"


class LayeredDiffusionBase:
    def __init__(self, model_file_name: str, model_url: str) -> None:
        self.model_file_name = model_file_name
        self.model_url = model_url

    def apply_c_concat(self, cond, uncond, c_concat):
        """Set foreground/background concat condition."""

        def write_c_concat(cond):
            new_cond = []
            for t in cond:
                n = [t[0], t[1].copy()]
                if "model_conds" not in n[1]:
                    n[1]["model_conds"] = {}
                n[1]["model_conds"]["c_concat"] = CONDRegular(c_concat)
                new_cond.append(n)
            return new_cond

        return (write_c_concat(cond), write_c_concat(uncond))

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        weight: float,
    ):
        """Patch model"""
        model_path = load_file_from_url(
            url=self.model_url,
            model_dir=layer_model_root,
            file_name=self.model_file_name,
        )
        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        layer_lora_patch_dict = to_lora_patch_dict(layer_lora_state_dict)
        work_model = model.clone()
        work_model.add_patches(layer_lora_patch_dict, weight)
        return (work_model,)


class LayeredDiffusionFG:
    """Generate foreground with transparent background."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "method": (
                    [
                        LayerMethod.ATTN.value,
                        LayerMethod.CONV.value,
                    ],
                    {
                        "default": LayerMethod.ATTN.value,
                    },
                ),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 3, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layered_diffusion"

    def __init__(self) -> None:
        self.fg_attn = LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_attn.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
        )
        self.fg_conv = LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_conv.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors",
        )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        method: str,
        weight: float,
    ):
        method = LayerMethod(method)
        if method == LayerMethod.ATTN:
            return self.fg_attn.apply_layered_diffusion(model, weight)
        if method == LayerMethod.CONV:
            return self.fg_conv.apply_layered_diffusion(model, weight)


class LayeredDiffusionCond:
    """Generate foreground + background given background / foreground."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cond": ("CONDITIONING",),
                "uncond": ("CONDITIONING",),
                "latent": ("LATENT",),
                "layer_type": (
                    [
                        LayerType.FG.value,
                        LayerType.BG.value,
                    ],
                    {
                        "default": LayerType.BG.value,
                    },
                ),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 3, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layered_diffusion"

    def __init__(self) -> None:
        self.fg_cond = LayeredDiffusionBase(
            model_file_name="layer_xl_fg2ble.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors",
        )
        self.bg_cond = LayeredDiffusionBase(
            model_file_name="layer_xl_bg2ble.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors",
        )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        cond,
        uncond,
        latent,
        layer_type,
        weight: float,
    ):
        layer_type = LayerType(layer_type)
        if layer_type == LayerType.FG:
            ld = self.fg_cond
        elif layer_type == LayerType.BG:
            ld = self.bg_cond

        c_concat = model.model.latent_format.process_in(latent["samples"])
        return ld.apply_layered_diffusion(model, weight) + ld.apply_c_concat(
            cond, uncond, c_concat
        )


NODE_CLASS_MAPPINGS = {
    "LayeredDiffusionApply": LayeredDiffusionFG,
    "LayeredDiffusionCondApply": LayeredDiffusionCond,
    "LayeredDiffusionDecode": LayeredDiffusionDecode,
    "LayeredDiffusionDecodeRGBA": LayeredDiffusionDecodeRGBA,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredDiffusionApply": "Layer Diffusion Apply",
    "LayeredDiffusionCondApply": "Layer Diffusion Cond Apply",
    "LayeredDiffusionDecode": "Layer Diffusion Decode",
    "LayeredDiffusionDecodeRGBA": "Layer Diffusion Decode (RGBA)",
}
