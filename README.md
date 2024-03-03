# ComfyUI-layerdiffusion
ComfyUI implementation of https://github.com/layerdiffusion/LayerDiffusion.

![image](https://github.com/huchenlei/ComfyUI-layerdiffusion/assets/20929282/413945a2-0948-405e-b524-e164ba54325d)

## Installation
run `pip install -r requirements.txt` to install python dependencies. You might experience version conflict on diffusers if you have other extensions
that depends on other versions of diffusers.

## Note
- `Apply Mask to Image` node comes from [comfyui-tooling-nodes](https://github.com/Acly/comfyui-tooling-nodes) package
- Currently only SDXL is supported. See https://github.com/layerdiffusion/sd-forge-layerdiffusion#model-notes for more details.

## TODO
- [ ] Foreground conditioning
- [ ] Background conditioning
