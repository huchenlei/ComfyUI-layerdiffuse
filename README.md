# ComfyUI-layerdiffuse
ComfyUI implementation of https://github.com/layerdiffusion/LayerDiffuse.

## Installation
Download the repository and unpack it into the custom_nodes folder in the ComfyUI installation directory.

Or clone via GIT, starting from ComfyUI installation directory:
```bash
cd custom_nodes
git clone git@github.com:huchenlei/ComfyUI-layerdiffuse.git
```

Run `pip install -r requirements.txt` to install python dependencies. You might experience version conflict on diffusers if you have other extensions that depend on other versions of diffusers. In this case, it is recommended to set up separate Python venvs.

## Workflows
### [Generate foreground](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_fg_example_rgba.json)
![rgba](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/5e6085e5-d997-4a0a-b589-257d65eb1eb2)

### [Generate foreground (RGB + alpha)](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_fg_example.json)
If you want more control of getting RGB images and alpha channel mask separately, you can use this workflow.
![readme1](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/4825b81c-7089-4806-bce7-777229421707)

### [Blending (FG/BG)](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_cond_example.json)
Blending given FG
![fg_cond](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/7f7dee80-6e57-4570-b304-d1f7e5dc3aad)

Blending given BG
![bg_cond](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/e3a79218-6123-453b-a54b-2f338db1c12d)

### [Extract FG from Blended + BG](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_diff_fg.json)
![diff_bg](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/45c7207d-72ff-4fb0-9c91-687040781837)

### [Extract BG from Blended + FG](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_diff_bg.json)
[Forge impl's sanity check](https://github.com/layerdiffuse/sd-forge-layerdiffuse#sanity-check) sets `Stop at` to 0.5 to get better quality BG.
This workflow might be inferior compared to other object removal workflows.
![diff_fg](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/05a10add-68b0-473a-acee-5853e4720322)

### [Extract BG from Blended + FG (Stop at 0.5)](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_diff_bg_stop_at.json)
In [SD Forge impl](https://github.com/layerdiffuse/sd-forge-layerdiffuse), there is a `stop at` param that determines when
layer diffuse should stop in the denoising process. In the background, what this param does is unapply the LoRA and c_concat cond after a certain step
threshold. This is hard/risky to implement directly in ComfyUI as it requires manually loading a model that has every change except the layer diffusion
change applied. A workaround in ComfyUI is to have another img2img pass on the layer diffuse result to simulate the effect of `stop at` param.
![diff_fg_stop_at](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/e383c9d3-2d47-40c2-b764-b0bd48243ee8)


### [Generate FG from BG combined](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_cond_fg_all.json)
Combines previous workflows to generate blended and FG given BG. We found that there are some color variations in the extracted FG. Need to confirm
with layer diffusion authors whether this is expected.
![fg_all](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/f4c18585-961a-473a-a616-aa3776bacd41)

### [2024-3-9] [Generate FG + Blended given BG](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_cond_joint_bg.json)
Need batch size = 2N. Currently only for SD15.
![sd15_cond_joint_bg](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/9bbfe5c1-14a0-421d-bf06-85e301bf8065)

### [2024-3-9] [Generate BG + Blended given FG](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_cond_joint_fg.json)
Need batch size = 2N. Currently only for SD15.
![sd15_cond_joint_fg](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/65af8b38-cf4c-4667-b76f-3013a0be0a48)

### [2024-3-9] [Generate BG + FG + Blended together](https://github.com/huchenlei/ComfyUI-layerdiffuse/blob/main/examples/layer_diffusion_joint.json)
Need batch size = 3N. Currently only for SD15.
![sd15_joint](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/e5545809-e3fb-4683-acf5-8728195cb2bc)

## Note
- Currently only SDXL/SD15 are supported. See https://github.com/layerdiffuse/sd-forge-layerdiffuse#model-notes for more details.
- To decode RGBA result, the generation dimension must be multiple of 64. Otherwise, you will get decode error: ![image](https://github.com/huchenlei/ComfyUI-layerdiffuse/assets/20929282/ff055f99-9297-4ff1-9a33-065aaadcf98e)
