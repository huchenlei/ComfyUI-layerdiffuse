# ComfyUI-layerdiffusion
ComfyUI implementation of https://github.com/layerdiffusion/LayerDiffusion.

## Installation
run `pip install -r requirements.txt` to install python dependencies. You might experience version conflict on diffusers if you have other extensions
that depends on other versions of diffusers.

## Workflows
### [Generate foreground](https://github.com/huchenlei/ComfyUI-layerdiffusion/blob/main/examples/layer_diffusion_fg_example_rgba.json)
![rgba](https://github.com/huchenlei/ComfyUI-layerdiffusion/assets/20929282/5e6085e5-d997-4a0a-b589-257d65eb1eb2)

### [Generate foreground (RGB + alpha)](https://github.com/huchenlei/ComfyUI-layerdiffusion/blob/main/examples/layer_diffusion_fg_example.json)
If you want more control of getting RGB image and alpha channel mask separately, you can use this workflow.
![readme1](https://github.com/huchenlei/ComfyUI-layerdiffusion/assets/20929282/4825b81c-7089-4806-bce7-777229421707)

### [Blending (FG/BG)](https://github.com/huchenlei/ComfyUI-layerdiffusion/blob/main/examples/layer_diffusion_cond_example.json)
Blending given FG
![fg_cond](https://github.com/huchenlei/ComfyUI-layerdiffusion/assets/20929282/7f7dee80-6e57-4570-b304-d1f7e5dc3aad)

Blending given BG
![bg_cond](https://github.com/huchenlei/ComfyUI-layerdiffusion/assets/20929282/e3a79218-6123-453b-a54b-2f338db1c12d)


## Note
- Currently only SDXL is supported. See https://github.com/layerdiffusion/sd-forge-layerdiffusion#model-notes for more details.

## TODO
- [x] Foreground conditioning
- [x] Background conditioning
- [ ] Blended + foreground => background
- [ ] Blended + background => foreground
