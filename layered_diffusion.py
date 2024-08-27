import os
from enum import Enum
import torch
import copy
from typing import Optional, List
from dataclasses import dataclass

import folder_paths
import comfy.model_management
import comfy.model_base
import comfy.supported_models
import comfy.supported_models_base
from comfy.model_patcher import ModelPatcher
from folder_paths import get_folder_paths
from comfy.utils import load_torch_file
from comfy_extras.nodes_compositing import JoinImageWithAlpha
from comfy.conds import CONDRegular
from .lib_layerdiffusion.utils import (
    load_file_from_url,
    to_lora_patch_dict,
)
from .lib_layerdiffusion.models import TransparentVAEDecoder
from .lib_layerdiffusion.attention_sharing import AttentionSharingPatcher
from .lib_layerdiffusion.enums import StableDiffusionVersion

if "layer_model" in folder_paths.folder_names_and_paths:
    layer_model_root = get_folder_paths("layer_model")[0]
else:
    layer_model_root = os.path.join(folder_paths.models_dir, "layer_model")
load_layer_model_state_dict = load_torch_file


class LayeredDiffusionDecode:
    """
    Decode alpha channel value from pixel value.
    [B, C=3, H, W] => [B, C=4, H, W]
    Outputs RGB image + Alpha mask.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "images": ("IMAGE",),
                "sd_version": (
                    [
                        StableDiffusionVersion.SD1x.value,
                        StableDiffusionVersion.SDXL.value,
                    ],
                    {
                        "default": StableDiffusionVersion.SDXL.value,
                    },
                ),
                "sub_batch_size": (
                    "INT",
                    {"default": 16, "min": 1, "max": 4096, "step": 1},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "decode"
    CATEGORY = "layer_diffuse"

    def __init__(self) -> None:
        self.vae_transparent_decoder = {}

    def decode(self, samples, images, sd_version: str, sub_batch_size: int):
        """
        sub_batch_size: How many images to decode in a single pass.
        See https://github.com/huchenlei/ComfyUI-layerdiffuse/pull/4 for more
        context.
        """
        sd_version = StableDiffusionVersion(sd_version)
        if sd_version == StableDiffusionVersion.SD1x:
            url = "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_vae_transparent_decoder.safetensors"
            file_name = "layer_sd15_vae_transparent_decoder.safetensors"
        elif sd_version == StableDiffusionVersion.SDXL:
            url = "https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/vae_transparent_decoder.safetensors"
            file_name = "vae_transparent_decoder.safetensors"

        if not self.vae_transparent_decoder.get(sd_version):
            model_path = load_file_from_url(
                url=url, model_dir=layer_model_root, file_name=file_name
            )
            self.vae_transparent_decoder[sd_version] = TransparentVAEDecoder(
                load_torch_file(model_path),
                device=comfy.model_management.get_torch_device(),
                dtype=(
                    torch.float16
                    if comfy.model_management.should_use_fp16()
                    else torch.float32
                ),
            )
        pixel = images.movedim(-1, 1)  # [B, H, W, C] => [B, C, H, W]

        # Decoder requires dimension to be 64-aligned.
        B, C, H, W = pixel.shape
        assert H % 64 == 0, f"Height({H}) is not multiple of 64."
        assert W % 64 == 0, f"Height({W}) is not multiple of 64."

        decoded = []
        for start_idx in range(0, samples["samples"].shape[0], sub_batch_size):
            decoded.append(
                self.vae_transparent_decoder[sd_version].decode_pixel(
                    pixel[start_idx : start_idx + sub_batch_size],
                    samples["samples"][start_idx : start_idx + sub_batch_size],
                )
            )
        pixel_with_alpha = torch.cat(decoded, dim=0)

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

    def decode(self, samples, images, sd_version: str, sub_batch_size: int):
        image, mask = super().decode(samples, images, sd_version, sub_batch_size)
        alpha = 1.0 - mask
        return JoinImageWithAlpha().join_image_with_alpha(image, alpha)


class LayeredDiffusionDecodeSplit(LayeredDiffusionDecodeRGBA):
    """Decode RGBA every N images."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT",),
                "images": ("IMAGE",),
                # Do RGBA decode every N output images.
                "frames": (
                    "INT",
                    {"default": 2, "min": 2, "max": s.MAX_FRAMES, "step": 1},
                ),
                "sd_version": (
                    [
                        StableDiffusionVersion.SD1x.value,
                        StableDiffusionVersion.SDXL.value,
                    ],
                    {
                        "default": StableDiffusionVersion.SDXL.value,
                    },
                ),
                "sub_batch_size": (
                    "INT",
                    {"default": 16, "min": 1, "max": 4096, "step": 1},
                ),
            },
        }

    MAX_FRAMES = 3
    RETURN_TYPES = ("IMAGE",) * MAX_FRAMES

    def decode(
        self,
        samples,
        images: torch.Tensor,
        frames: int,
        sd_version: str,
        sub_batch_size: int,
    ):
        sliced_samples = copy.copy(samples)
        sliced_samples["samples"] = sliced_samples["samples"][::frames]
        return tuple(
            (
                (
                    super(LayeredDiffusionDecodeSplit, self).decode(
                        sliced_samples, imgs, sd_version, sub_batch_size
                    )[0]
                    if i == 0
                    else imgs
                )
                for i in range(frames)
                for imgs in (images[i::frames],)
            )
        ) + (None,) * (self.MAX_FRAMES - frames)


class LayerMethod(Enum):
    ATTN = "Attention Injection"
    CONV = "Conv Injection"


class LayerType(Enum):
    FG = "Foreground"
    BG = "Background"


@dataclass
class LayeredDiffusionBase:
    model_file_name: str
    model_url: str
    sd_version: StableDiffusionVersion
    attn_sharing: bool = False
    injection_method: Optional[LayerMethod] = None
    cond_type: Optional[LayerType] = None
    # Number of output images per run.
    frames: int = 1

    @property
    def config_string(self) -> str:
        injection_method = self.injection_method.value if self.injection_method else ""
        cond_type = self.cond_type.value if self.cond_type else ""
        attn_sharing = "attn_sharing" if self.attn_sharing else ""
        frames = f"Batch size ({self.frames}N)" if self.frames != 1 else ""
        return ", ".join(
            x
            for x in (
                self.sd_version.value,
                injection_method,
                cond_type,
                attn_sharing,
                frames,
            )
            if x
        )

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
        def pad_diff_weight(v):
            if len(v) == 1:
                return ("diff", [v[0], {"pad_weight": True}])
            elif len(v) == 2 and v[0] == "diff":
                return ("diff", [v[1][0], {"pad_weight": True}])
            else:
                return v

        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        layer_lora_patch_dict = {
            k: pad_diff_weight(v)
            for k, v in to_lora_patch_dict(layer_lora_state_dict).items()
        }
        work_model = model.clone()
        work_model.add_patches(layer_lora_patch_dict, weight)
        return (work_model,)

    def apply_layered_diffusion_attn_sharing(
        self,
        model: ModelPatcher,
        control_img: Optional[torch.TensorType] = None,
    ):
        """Patch model with attn sharing"""
        model_path = load_file_from_url(
            url=self.model_url,
            model_dir=layer_model_root,
            file_name=self.model_file_name,
        )
        layer_lora_state_dict = load_layer_model_state_dict(model_path)
        work_model = model.clone()
        patcher = AttentionSharingPatcher(
            work_model, self.frames, use_control=control_img is not None
        )
        patcher.load_state_dict(layer_lora_state_dict, strict=True)
        if control_img is not None:
            patcher.set_control(control_img)
        return (work_model,)


def get_model_sd_version(model: ModelPatcher) -> StableDiffusionVersion:
    """Get model's StableDiffusionVersion."""
    base: comfy.model_base.BaseModel = model.model
    model_config: comfy.supported_models.supported_models_base.BASE = base.model_config
    if isinstance(model_config, comfy.supported_models.SDXL):
        return StableDiffusionVersion.SDXL
    elif isinstance(
        model_config, (comfy.supported_models.SD15, comfy.supported_models.SD20)
    ):
        # SD15 and SD20 are compatible with each other.
        return StableDiffusionVersion.SD1x
    else:
        raise Exception(f"Unsupported SD Version: {type(model_config)}.")


class LayeredDiffusionFG:
    """Generate foreground with transparent background."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "config": ([c.config_string for c in s.MODELS],),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 3, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layer_diffuse"
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_attn.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_attn.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            injection_method=LayerMethod.ATTN,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_transparent_conv.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_transparent_conv.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            injection_method=LayerMethod.CONV,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_sd15_transparent_attn.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_transparent_attn.safetensors",
            sd_version=StableDiffusionVersion.SD1x,
            injection_method=LayerMethod.ATTN,
            attn_sharing=True,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        config: str,
        weight: float,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        if ld_model.attn_sharing:
            return ld_model.apply_layered_diffusion_attn_sharing(model)
        else:
            return ld_model.apply_layered_diffusion(model, weight)


class LayeredDiffusionJoint:
    """Generate FG + BG + Blended in one inference batch. Batch size = 3N."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "config": ([c.config_string for c in s.MODELS],),
            },
            "optional": {
                "fg_cond": ("CONDITIONING",),
                "bg_cond": ("CONDITIONING",),
                "blended_cond": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layer_diffuse"
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_sd15_joint.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_joint.safetensors",
            sd_version=StableDiffusionVersion.SD1x,
            attn_sharing=True,
            frames=3,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        config: str,
        fg_cond: Optional[List[List[torch.TensorType]]] = None,
        bg_cond: Optional[List[List[torch.TensorType]]] = None,
        blended_cond: Optional[List[List[torch.TensorType]]] = None,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        assert ld_model.attn_sharing
        work_model = ld_model.apply_layered_diffusion_attn_sharing(model)[0]
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["cond_overwrite"] = [
            cond[0][0] if cond is not None else None
            for cond in (
                fg_cond,
                bg_cond,
                blended_cond,
            )
        ]
        return (work_model,)


class LayeredDiffusionCond:
    """Generate foreground + background given background / foreground.
    - FG => Blended
    - BG => Blended
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cond": ("CONDITIONING",),
                "uncond": ("CONDITIONING",),
                "latent": ("LATENT",),
                "config": ([c.config_string for c in s.MODELS],),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 3, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layer_diffuse"
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_fg2ble.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fg2ble.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_bg2ble.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bg2ble.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        cond,
        uncond,
        latent,
        config: str,
        weight: float,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        c_concat = model.model.latent_format.process_in(latent["samples"])
        return ld_model.apply_layered_diffusion(
            model, weight
        ) + ld_model.apply_c_concat(cond, uncond, c_concat)


class LayeredDiffusionCondJoint:
    """Generate fg/bg + blended given fg/bg.
    - FG => Blended + BG
    - BG => Blended + FG
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "config": ([c.config_string for c in s.MODELS],),
            },
            "optional": {
                "cond": ("CONDITIONING",),
                "blended_cond": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layer_diffuse"
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_sd15_fg2bg.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_fg2bg.safetensors",
            sd_version=StableDiffusionVersion.SD1x,
            attn_sharing=True,
            frames=2,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_sd15_bg2fg.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_sd15_bg2fg.safetensors",
            sd_version=StableDiffusionVersion.SD1x,
            attn_sharing=True,
            frames=2,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        image,
        config: str,
        cond: Optional[List[List[torch.TensorType]]] = None,
        blended_cond: Optional[List[List[torch.TensorType]]] = None,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        assert ld_model.attn_sharing
        work_model = ld_model.apply_layered_diffusion_attn_sharing(
            model, control_img=image.movedim(-1, 1)
        )[0]
        work_model.model_options.setdefault("transformer_options", {})
        work_model.model_options["transformer_options"]["cond_overwrite"] = [
            cond[0][0] if cond is not None else None
            for cond in (
                cond,
                blended_cond,
            )
        ]
        return (work_model,)


class LayeredDiffusionDiff:
    """Extract FG/BG from blended image.
    - Blended + FG => BG
    - Blended + BG => FG
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "cond": ("CONDITIONING",),
                "uncond": ("CONDITIONING",),
                "blended_latent": ("LATENT",),
                "latent": ("LATENT",),
                "config": ([c.config_string for c in s.MODELS],),
                "weight": (
                    "FLOAT",
                    {"default": 1.0, "min": -1, "max": 3, "step": 0.05},
                ),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING")
    FUNCTION = "apply_layered_diffusion"
    CATEGORY = "layer_diffuse"
    MODELS = (
        LayeredDiffusionBase(
            model_file_name="layer_xl_fgble2bg.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_fgble2bg.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            cond_type=LayerType.FG,
        ),
        LayeredDiffusionBase(
            model_file_name="layer_xl_bgble2fg.safetensors",
            model_url="https://huggingface.co/LayerDiffusion/layerdiffusion-v1/resolve/main/layer_xl_bgble2fg.safetensors",
            sd_version=StableDiffusionVersion.SDXL,
            cond_type=LayerType.BG,
        ),
    )

    def apply_layered_diffusion(
        self,
        model: ModelPatcher,
        cond,
        uncond,
        blended_latent,
        latent,
        config: str,
        weight: float,
    ):
        ld_model = [m for m in self.MODELS if m.config_string == config][0]
        assert get_model_sd_version(model) == ld_model.sd_version
        c_concat = model.model.latent_format.process_in(
            torch.cat([latent["samples"], blended_latent["samples"]], dim=1)
        )
        return ld_model.apply_layered_diffusion(
            model, weight
        ) + ld_model.apply_c_concat(cond, uncond, c_concat)


NODE_CLASS_MAPPINGS = {
    "LayeredDiffusionApply": LayeredDiffusionFG,
    "LayeredDiffusionJointApply": LayeredDiffusionJoint,
    "LayeredDiffusionCondApply": LayeredDiffusionCond,
    "LayeredDiffusionCondJointApply": LayeredDiffusionCondJoint,
    "LayeredDiffusionDiffApply": LayeredDiffusionDiff,
    "LayeredDiffusionDecode": LayeredDiffusionDecode,
    "LayeredDiffusionDecodeRGBA": LayeredDiffusionDecodeRGBA,
    "LayeredDiffusionDecodeSplit": LayeredDiffusionDecodeSplit,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LayeredDiffusionApply": "Layer Diffuse Apply",
    "LayeredDiffusionJointApply": "Layer Diffuse Joint Apply",
    "LayeredDiffusionCondApply": "Layer Diffuse Cond Apply",
    "LayeredDiffusionCondJointApply": "Layer Diffuse Cond Joint Apply",
    "LayeredDiffusionDiffApply": "Layer Diffuse Diff Apply",
    "LayeredDiffusionDecode": "Layer Diffuse Decode",
    "LayeredDiffusionDecodeRGBA": "Layer Diffuse Decode (RGBA)",
    "LayeredDiffusionDecodeSplit": "Layer Diffuse Decode (Split)",
}
