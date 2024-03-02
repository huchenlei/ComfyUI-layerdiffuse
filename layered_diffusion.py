import os
from enum import Enum
import torch
import json
import random
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

import folder_paths
import comfy.model_management
from nodes import SaveImage
from comfy.cli_args import args
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


class SaveRGBAImage(SaveImage):
    def save_images(
        self, images, filename_prefix="ComfyUI", prompt=None, extra_pnginfo=None
    ):
        alpha = images[..., :1]
        fg = images[..., 1:]
        pngs = torch.cat([fg, alpha], dim=3)
        pngs = (
            (pngs * 255.0).detach().cpu().float().numpy().clip(0, 255).astype(np.uint8)
        )

        filename_prefix += self.prefix_append
        full_output_folder, filename, counter, subfolder, filename_prefix = (
            folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, images[0].shape[1], images[0].shape[0]
            )
        )
        results = list()
        for batch_number, image in enumerate(pngs):
            img = Image.fromarray(image)
            metadata = None
            if not args.disable_metadata:
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

            filename_with_batch_num = filename.replace("%batch_num%", str(batch_number))
            file = f"{filename_with_batch_num}_{counter:05}_.png"
            img.save(
                os.path.join(full_output_folder, file),
                pnginfo=metadata,
                compress_level=self.compress_level,
            )
            results.append(
                {"filename": file, "subfolder": subfolder, "type": self.type}
            )
            counter += 1

        return {"ui": {"images": results}}


class PreviewRGBAImage(SaveRGBAImage):
    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + "".join(
            random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5)
        )
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }


class LayeredDiffusionDecode:
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    """

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"samples": ("LATENT",), "images": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
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
        return (pixel_with_alpha,)


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
    "SaveRGBAImage": SaveRGBAImage,
    "PreviewRGBAImage": PreviewRGBAImage,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredDiffusionApply": "Layered Diffusion Apply",
    "LayeredDiffusionDecode": "Layered Diffusion Decode",
    "SaveRGBAImage": "Save RGBA Image",
    "PreviewRGBAImage": "Preview RGBA Image",
}
