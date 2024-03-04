import os
from enum import Enum
import torch

import folder_paths
import comfy.model_management
from comfy.model_patcher import ModelPatcher
from comfy.utils import load_torch_file
from .lib_layerdiffusion.utils import (
    rgba2rgbfp32,
    load_file_from_url,
    to_lora_patch_dict,
)
from .lib_layerdiffusion.models import TransparentVAEDecoder


layer_model_root = os.path.join(folder_paths.models_dir, "layer_model")
load_layer_model_state_dict = load_torch_file


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
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                {
                "samples": ("LATENT",), 
                "images": ("IMAGE",),
                "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
                },
            }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "decode"
    CATEGORY = "layered_diffusion"

    def __init__(self) -> None:
        self.vae_transparent_decoder = None

    def decode(self, samples, images, per_batch):    
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
        pixel = images.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]
        decoded = []
        for start_idx in range(0, samples["samples"].shape[0], per_batch):
            decoded.append(self.vae_transparent_decoder.decode_pixel(pixel[start_idx:start_idx+per_batch], samples["samples"][start_idx:start_idx+per_batch]))
        pixel_with_alpha = torch.cat(decoded, dim=0)
        
        # [B, C, H, W] => [B, H, W, C]
        pixel_with_alpha = pixel_with_alpha.movedim(1, -1)
        image = pixel_with_alpha[..., 1:]
        alpha = pixel_with_alpha[..., 0]
        return (image, alpha,)


class LayerMethod(Enum):
    ATTN = "Attention Injection"
    CONV = "Conv Injection"


class LayeredDiffusionApply:
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

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        method: str,
        weight: float,
    ):
        """Patch model"""
        method = LayerMethod(method)

        # Patch unet
        if method == LayerMethod.ATTN:
            model_path = load_file_from_url(
                url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
                model_dir=layer_model_root,
                file_name="layer_xl_transparent_attn.safetensors",
            )
        if method == LayerMethod.CONV:
            model_path = load_file_from_url(
                url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors",
                model_dir=layer_model_root,
                file_name="layer_xl_transparent_conv.safetensors",
            )

        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        layer_lora_patch_dict = to_lora_patch_dict(layer_lora_state_dict)
        work_model = model.clone()
        work_model.add_patches(layer_lora_patch_dict, weight)
        return (work_model,)


NODE_CLASS_MAPPINGS = {
    "LayeredDiffusionApply": LayeredDiffusionApply,
    "LayeredDiffusionDecode": LayeredDiffusionDecode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredDiffusionApply": "Layer Diffusion Apply",
    "LayeredDiffusionDecode": "Layer Diffusion Decode",
}
