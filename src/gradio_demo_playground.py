import os
import re
import time
from io import BytesIO
import uuid
from dataclasses import dataclass
from glob import iglob
import argparse
from einops import rearrange
#from fire import Fire
from PIL import ExifTags, Image
from safetensors.torch import load_file, save_file


import torch
import torch.nn.functional as F
import gradio as gr
import numpy as np
from transformers import pipeline

from flux.sampling import denoise_fireflow, get_schedule, prepare, prepare_image, unpack, denoise_rf, denoise_rf_solver, denoise_midpoint, denoise_rf_inversion, denoise_multi_turn_consistent, get_noise
from flux.util import (configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5)

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

@dataclass
class SamplingOptions:
    source_prompt: str
    target_prompt: str
    # prompt: str
    width: int
    height: int
    num_steps: int
    guidance: float
    seed: int | None

@torch.inference_mode()
def encode(init_image, torch_device, ae):
    init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 127.5 - 1
    init_image = init_image.unsqueeze(0) 
    init_image = init_image.to(torch_device)
    with torch.no_grad():
        init_image = ae.encode(init_image).to(torch.bfloat16)
    return init_image


class FluxEditor:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.offload = args.offload
        self.name = args.name
        self.is_schnell = args.name == "flux-schnell"

        self.feature_path = 'feature'

        out_root = 'src/gradio_utils/gradio_outputs'
        name_dir = f'exp_{len(os.listdir(out_root))}'
        self.output_dir = os.path.join(out_root, name_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self.instructions = ['source']
        self.source_image = None
        self.history_tensors = {
            "source img": torch.zeros((1, 1, 1)),
            "prev img": torch.zeros((1, 1, 1))}   

        self.add_sampling_metadata = True

        if self.name not in configs:
            available = ", ".join(configs.keys())
            raise ValueError(f"Got unknown model name: {self.name}, chose from {available}")

        # init all components
        self.clip = load_clip(self.device)
        self.t5 = load_t5(self.device, max_length=256 if self.name == "flux-schnell" else 512)
        self.model = load_flow_model(self.name, device="cpu" if self.offload else self.device)
        self.ae = load_ae(self.name, device="cpu" if self.offload else self.device)
        self.t5.eval()
        self.clip.eval()
        self.ae.eval()
        self.model.eval()

        # clear history
        if os.path.exists("history_gradio/history.safetensors"):
            os.remove("history_gradio/history.safetensors")
    

    @torch.inference_mode()
    def process_image(self, 
                    init_image,
                    source_prompt, 
                    target_prompt, 
                    editing_strategy, 
                    denoise_strategy, 
                    num_steps, 
                    guidance, 
                    attn_guidance_start_block, 
                    inject_step, 
                    init_image_2=None):
        if init_image is None:
            img, gr_gallery = self.generate_image(prompt=target_prompt)
        else:
            img, gr_gallery = self.edit(init_image, source_prompt, target_prompt, editing_strategy, denoise_strategy, num_steps, guidance, attn_guidance_start_block, inject_step, init_image_2)
        return img, gr_gallery

        

    @torch.inference_mode()
    def generate_image(
        self,
        width=512,
        height=512,
        num_steps=28,
        guidance=3.5,
        seed=None,
        prompt='',
        init_image=None,
        image2image_strength=0.0,
        add_sampling_metadata=True,
    ):

        if seed is None:
            g_seed = torch.Generator(device="cpu").seed()
        print(f"Generating '{prompt}' with seed {g_seed}")
        t0 = time.perf_counter()

        if init_image is not None:
            if isinstance(init_image, np.ndarray):
                init_image = torch.from_numpy(init_image).permute(2, 0, 1).float() / 255.0
                init_image = init_image.unsqueeze(0)
            init_image = init_image.to(self.device)
            init_image = torch.nn.functional.interpolate(init_image, (height, width))
            if self.offload:
                self.ae.encoder.to(self.device)
            init_image = self.ae.encode(init_image)
            if self.offload:
                self.ae = self.ae.cpu()
                torch.cuda.empty_cache()

        # prepare input
        x = get_noise(
            1,
            height,
            width,
            device=self.device,
            dtype=torch.bfloat16,
            seed=g_seed,
        )
        timesteps = get_schedule(
            num_steps,
            x.shape[-1] * x.shape[-2] // 4,
            shift=(not self.is_schnell),
        )
        if init_image is not None:
            t_idx = int((1 - image2image_strength) * num_steps)
            t = timesteps[t_idx]
            timesteps = timesteps[t_idx:]
            x = t * x + (1.0 - t) * init_image.to(x.dtype)

        if self.offload:
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)
        inp = prepare(t5=self.t5, clip=self.clip, img=x, prompt=prompt)

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)

        # denoise initial noise
        info = {}
        info['feature'] = {}
        info['inject_step'] = 0
        info['editing_strategy']= ""
        info['start_layer_index'] = 0
        info['end_layer_index'] = 37
        info['reuse_v']= False
        qkv_ratio = '1.0,1.0,1.0'
        info['qkv_ratio'] = list(map(float, qkv_ratio.split(',')))
        x = denoise_rf(self.model, **inp, timesteps=timesteps, guidance=guidance, inverse=False, info=info)

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)

        # decode latents to pixel space
        x = unpack(x[0].float(), height, width)
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if self.offload:
            self.ae.decoder.cpu()
            torch.cuda.empty_cache()

        t1 = time.perf_counter()

        print(f"Done in {t1 - t0:.1f}s.")
        # bring into PIL format
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        
        filename = os.path.join(self.output_dir,f"round_0000_[{prompt}].jpg")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        exif_data = Image.Exif()
        if init_image is None:
            exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        else:
            exif_data[ExifTags.Base.Software] = "AI generated;img2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = prompt
        img.save(filename, format="jpeg", exif=exif_data, quality=95, subsampling=0)
        self.instructions = [prompt]

            #--------------------    6.4 save editing prompt, update gradio component: gallery      ----------------------#
        img_and_prompt = []
        history_imgs = sorted(os.listdir(self.output_dir))
        for img_file, prompt_txt in zip(history_imgs, self.instructions):
            img_and_prompt.append((os.path.join(self.output_dir, img_file), prompt_txt))
            history_gallery = gr.Gallery(value=img_and_prompt, label="History Image", interactive=True, columns=3)
        return img, history_gallery


    @torch.inference_mode()
    def edit(self, init_image, source_prompt, target_prompt, editing_strategy, denoise_strategy, num_steps, guidance, attn_guidance_start_block, inject_step, init_image_2=None):
        
        torch.cuda.empty_cache()
        seed = None
        
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.encoder.to(self.device)

        #-----------------------------     0.1 prepare multi-turn editing     -------------------------------------#
        info = {}
        shape = init_image.shape
        new_h = shape[0] if shape[0] % 16 == 0 else shape[0] - shape[0] % 16
        new_w = shape[1] if shape[1] % 16 == 0 else shape[1] - shape[1] % 16

        if not any("round_0000" in fname for fname in os.listdir(self.output_dir)):
            Image.fromarray(init_image).save(os.path.join(self.output_dir,"round_0000_[source].jpg"))


        init_image = init_image[:new_h, :new_w, :]
        width, height = init_image.shape[0], init_image.shape[1]
        init_image = encode(init_image, self.device, self.ae)

        print(init_image.shape)

        if init_image_2 is None:
            print("init_image_2 is not provided, proceeding with single image processing.")
        else: 
            init_image_2_pil = Image.fromarray(init_image_2) # Convert NumPy array to PIL Image
            init_image_2_pil = init_image_2_pil.resize((new_w, new_h), Image.Resampling.LANCZOS) 
            init_image_2 = np.array(init_image_2_pil)  # Convert back to NumPy (if needed)
            init_image_2 = encode(init_image_2, self.device, self.ae)

        rng = torch.Generator(device="cpu")
        opts = SamplingOptions(
            source_prompt=source_prompt,
            target_prompt=target_prompt,
            width=width,
            height=height,
            num_steps=num_steps,
            guidance=guidance,
            seed=seed,
        )
        if opts.seed is None:
            opts.seed = torch.Generator(device="cpu").seed()
        
        print(f"Editing with prompt:\n{opts.source_prompt}")
        t0 = time.perf_counter()

        opts.seed = None
        if self.offload:
            self.ae = self.ae.cpu()
            torch.cuda.empty_cache()
            self.t5, self.clip = self.t5.to(self.device), self.clip.to(self.device)

        #-----------------------------     0.2 prepare attention strategy     -------------------------------------#
        info = {}
        info['feature'] = {}
        info['inject_step'] = inject_step
        info['editing_strategy']= " ".join(editing_strategy)
        info['start_layer_index'] = 0
        info['end_layer_index'] = 37
        info['reuse_v']= False
        qkv_ratio = '1.0,1.0,1.0'
        info['qkv_ratio'] = list(map(float, qkv_ratio.split(',')))
        info['attn_guidance'] = attn_guidance_start_block
        info['lqr_stop'] = 0.25

        if not os.path.exists(self.feature_path):
            os.mkdir(self.feature_path)


        #-----------------------------     0.3 prepare latents     -------------------------------------#
        with torch.no_grad():
            inp = prepare(self.t5, self.clip, init_image, prompt=opts.source_prompt)
            inp_target = prepare(self.t5, self.clip, init_image, prompt=opts.target_prompt)
            if self.source_image is None:
                self.source_image = inp['img']
            inp_target_2 = None
            if not init_image_2 is None:
                inp_target_2 = prepare_image(init_image_2)
                info['lqr_stop'] = 0.35

        timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=(self.name != "flux-schnell"))
        #timesteps = get_schedule(opts.num_steps, inp["img"].shape[1], shift=False)

        # offload TEs to CPU, load model to gpu
        if self.offload:
            self.t5, self.clip = self.t5.cpu(), self.clip.cpu()
            torch.cuda.empty_cache()
            self.model = self.model.to(self.device)


        
        #-----------------------------     1 Inverting current image     -------------------------------------#
        denoise_strategies = ['fireflow', 'rf', 'rf_solver', 'midpoint', 'rf_inversion', 'multi_turn_consistent']
        denoise_funcs = [denoise_fireflow, denoise_rf, denoise_rf_solver, denoise_midpoint, denoise_rf_inversion, denoise_multi_turn_consistent]
        denoise_func = denoise_funcs[denoise_strategies.index(denoise_strategy)]
        with torch.no_grad():
            z, info = denoise_func(self.model, **inp, timesteps=timesteps, guidance=1, inverse=True, info=info)
        
        
        
        
        #-----------------------------     2 history_tensors used to implement dual-LQR guiding editing     -------------------------------------#
        inp_target["img"] = z
        timesteps = get_schedule(opts.num_steps, inp_target["img"].shape[1], shift=(self.name != "flux-schnell"))

        if torch.all(self.history_tensors['source img'] == 0):
            self.history_tensors = {
            "source img": inp["img"],
            "prev img": inp_target_2} 
        else:
            if inp_target_2 is None:
                self.history_tensors["prev img"] = inp["img"]
            else:
                self.history_tensors["source img"] = inp["img"]
                self.history_tensors["prev img"] = inp_target_2

        #-----------------------------     3 sampling     -------------------------------------#
        if denoise_strategy in ['rf_inversion', 'multi_turn_consistent']:
            x, _ = denoise_func(self.model, **inp_target, timesteps=timesteps, guidance=guidance, inverse=False, info=info, img_LQR=self.history_tensors)
        else:
            x, _ = denoise_func(self.model, **inp_target, timesteps=timesteps, guidance=opts.guidance, inverse=False, info=info)
        

        #-----------------------------     4 update history_tensors     -------------------------------------#
        info = {}
        self.history_tensors["source img"] = self.source_image
        self.history_tensors["prev img"] = x
        '''save_file(history_tensors, "history_gradio/history.safetensors")'''

        # offload model, load autoencoder to gpu
        if self.offload:
            self.model.cpu()
            torch.cuda.empty_cache()
            self.ae.decoder.to(x.device)



        #-----------------------------     5 decode x to image      -------------------------------------#
        x = unpack(x.float(), opts.width, opts.height)

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            x = self.ae.decode(x)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        # bring into PIL format and save
        x = x.clamp(-1, 1)
        x = embed_watermark(x.float())
        x = rearrange(x[0], "c h w -> h w c")

        img = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        exif_data = Image.Exif()
        exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
        exif_data[ExifTags.Base.Make] = "Black Forest Labs"
        exif_data[ExifTags.Base.Model] = self.name
        if self.add_sampling_metadata:
            exif_data[ExifTags.Base.ImageDescription] = source_prompt
    


        #--------------------------------     6 save image      -------------------------------------#

        #--------------------     6.1 prepare output folder      ----------------------#
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            idx = 1
        #--------------------    6.2 editing round      ----------------------#           
        else:
            fns = [fn for fn in os.listdir(self.output_dir)]
            if len(fns) > 0:
                idx = max(int(fn.split("_")[1]) for fn in fns) + 1
            else:
                idx = 1
        formatted_idx = str(idx).zfill(4) # Format as a 4-digit string

        #--------------------    6.3 output name      ----------------------#
        if denoise_strategy == 'multi_turn_consistent':
            denoise_strategy = 'MTC'
        if target_prompt == '':
            target_prompt = 'Reconstruction'
        if target_prompt == source_prompt:
            target_prompt = 'Reconstruction: ' + target_prompt

        output_name = f"round_{formatted_idx}_[{" ".join(target_prompt.split()[-5:])}]_{denoise_strategy}.jpg"
        fn = os.path.join(self.output_dir, output_name)
        
        print(f"Done in {t1 - t0:.1f}s. Saving {fn}")
        img.save(fn)

        if 'Reconstruction' in target_prompt:
            target_prompt = source_prompt
        self.instructions.append(target_prompt)
        print("End Edit")

        #--------------------    6.4 save editing prompt, update gradio component: gallery      ----------------------#
        img_and_prompt = []
        history_imgs = sorted(os.listdir(self.output_dir))
        for img_file, prompt_txt in zip(history_imgs, self.instructions):
            img_and_prompt.append((os.path.join(self.output_dir, img_file), prompt_txt))
        history_gallery = gr.Gallery(value=img_and_prompt, label="History Image", interactive=True, columns=3)
    
        return img, history_gallery


def on_select(gallery, selected: gr.SelectData):
    return gallery[selected.index][0], gallery[selected.index][1]

def on_upload(path, uploaded: gr.EventData):
    return path[0][0]

def reset():
    source_prompt = "(Optional) Describe the content of the uploaded image."
    traget_prompt = "(Required) Describe the desired content of the edited image."
    gallery = None
    output_image = None
    return source_prompt, traget_prompt, gallery, output_image

def create_demo(model_name: str, device: str = "cuda" if torch.cuda.is_available() else "cpu", offload: bool = False):
    editor = FluxEditor(args)
    is_schnell = model_name == "flux-schnell"
    
    # Pre-defined examples
    examples = [
        ["gradio_examples/dog.jpg", "Photograph of a dog on the grass", "Photograph of a cat on the grass", ['replace_v'], 8, 1, 2.0],
        ["gradio_examples/gold.jpg", "3d melting gold render", "a cat in the style of 3d melting gold render", ['replace_v'], 8, 1, 2.0],
        ["gradio_examples/gold.jpg", "3d melting gold render", "a cat in the style of 3d melting gold render", ['replace_v'], 10, 1, 2.0],
        ["gradio_examples/boy.jpg", "A young boy is playing with a toy airplane on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above.", "A young boy is sitting on the grassy front lawn of a suburban house, with a blue sky and fluffy clouds above.", ['replace_v'], 8, 1, 2.0],
        ["gradio_examples/cartoon.jpg", "", "a cartoon style Albert Einstein raising his left hand", ['replace_v'], 8, 1, 2.0],
        ["gradio_examples/cartoon.jpg", "", "a cartoon style Albert Einstein raising his left hand", ['replace_v'], 10, 1, 2.0],
        ["gradio_examples/cartoon.jpg", "", "a cartoon style Albert Einstein raising his left hand", ['replace_v'], 15, 1, 2.0],
        ["gradio_examples/art.jpg", "", "a vivid depiction of the Batman, featuring rich, dynamic colors,  and a blend of realistic and abstract elements with dynamic splatter art.", ['add_q'], 8, 1, 2.0],
    ]

    with gr.Blocks() as demo:
        gr.Markdown(f"# Multi-turn Consistent Image Editing (FLUX.1-dev)")
        
        with gr.Row():
            with gr.Column():
                source_prompt = gr.Textbox(label="Source Prompt", value="(Optional) Describe the content of the uploaded image.")
                target_prompt = gr.Textbox(label="Target Prompt", value="(Required) Describe the desired content of the edited image.")
                with gr.Row():
                    init_image = gr.Image(label="Input Image", visible=False, width=200)
                    init_image_2 = gr.Image(label="Input Image 2", visible=False, width=200)
                gallery = gr.Gallery(label ="History Image", interactive=True, columns=3)
                editing_strategy = gr.CheckboxGroup(
                    label="Editing Technique",
                    choices=['attn_guidance', 'replace_v', 'add_q', 'add_k', 'add_v', 'replace_q', 'replace_k'],
                    value=['attn_guidance'],  # Default: none selected
                    interactive=True
                )
                denoise_strategy = gr.Dropdown(
                    ['multi_turn_consistent', 'fireflow', 'rf', 'rf_solver', 'midpoint', 'rf_inversion'], 
                    label="Denoising Technique", value='multi_turn_consistent')
                generate_btn = gr.Button("Generate")
            
            with gr.Column():
                with gr.Accordion("Advanced Options", open=True):
                    num_steps = gr.Slider(1, 30, 15, step=1, label="Number of steps")
                    guidance = gr.Slider(1.0, 10.0, 3.5, step=0.1, label="Text Guidance", interactive=not is_schnell)
                    attn_guidance_start_block = gr.Slider(0, 18, 11, step=1, label="Top activated attn-maps", interactive=not is_schnell)
                    inject_step = gr.Slider(0, 15, 1, step=1, label="Number of inject steps")
                output_image = gr.Image(label="Generated/Edited Image")
                reset_btn = gr.Button("Reset")

        gallery.select(on_select, gallery, [init_image, source_prompt])
        gallery.upload(on_upload, gallery, init_image)
        generate_btn.click(
            fn=editor.process_image,
            inputs=[init_image, source_prompt, target_prompt, editing_strategy, denoise_strategy, num_steps, guidance, attn_guidance_start_block, inject_step, init_image_2],
            outputs=[output_image, gallery]
        )
        reset_btn.click(fn = reset, outputs=[source_prompt, target_prompt, gallery, output_image])
        
        # Add examples
        gr.Examples(
            examples=examples,
            inputs=[
                init_image, 
                source_prompt, 
                target_prompt, 
                editing_strategy, 
                num_steps, 
                inject_step, 
                guidance
            ],
            outputs=[output_image],
            fn=editor.edit,
        )


    return demo


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Flux")
    parser.add_argument("--name", type=str, default="flux-dev", choices=list(configs.keys()), help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--offload", action="store_true", help="Offload model to CPU when not in use")
    parser.add_argument("--share", action="store_true", help="Create a public link to your demo")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    demo = create_demo(args.name, args.device, args.offload)
    #demo.launch(server_name='0.0.0.0', share=args.share, server_port=args.port)
    demo.launch(share=True)
