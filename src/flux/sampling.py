import math
from typing import Callable, Optional, Union, List, Dict, Any
import os
from PIL import Image

import torch
from einops import rearrange, repeat
from torch import Tensor

from .model import Flux
from .modules.conditioner import HFEmbedder
from .modules.autoencoder import AutoEncoder



def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_image(img: Tensor):
    bs, c, h, w = img.shape
    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    return img


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )



def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise_rf(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step = i
        )
        img = img + (t_prev - t_curr) * pred

    return img, info


def denoise_rf_solver(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    img_ori: Optional[Tensor] = None
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step = i
        )

        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), (t_curr + (t_prev - t_curr) / 2), dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            cur_step = i
        )

        first_order = (pred_mid - pred) / ((t_prev - t_curr) / 2)
        img = img + (t_prev - t_curr) * pred + 0.5 * (t_prev - t_curr) ** 2 * first_order

    return img, info


def denoise_fireflow(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    img_ori: Optional[Tensor] = None,
    ae: Optional[AutoEncoder] = None,  # Optional AutoEncoder for decoding
    device: Optional[Union[str, torch.device]] = None  # Optional device specification
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
                cur_step=i
            )
        else:
            pred = next_step_velocity
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

###########################         save generating steps         ##############################
        #idx = len(timesteps) - 1
        #fn = f'result/intermediate_{idx}steps'
        #if not os.path.exists(fn):
            #os.makedirs(fn)
        #fn += f'/fireflow_{t_prev}.jpg'
        #if inverse:
            #fn = f'result/intermediate_{idx}steps/inverse_fireflow_{t_prev}.jpg'

        # decode latents to pixel space
        #x = unpack(img.float(), img.shape[1] ** 0.5 * 16, img.shape[1] ** 0.5 * 16)

        #with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            #x = ae.decode(x)
        
        # bring into PIL format and save
        #x = x.clamp(-1, 1)
        #x = rearrange(x[0], "c h w -> h w c")
        #x = Image.fromarray((127.5 * (x + 1.0)).cpu().byte().numpy())
        #x.save(fn)
###########################         save generating steps         ##############################       

    return img, info


def denoise_midpoint(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    if inverse:
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info
        )
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info
        )
        next_step_velocity = pred_mid
        
        img = img + (t_prev - t_curr) * pred_mid

    return img, info


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


def denoise_rf_inversion(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    stop_timestep: float = 0.35,
    img_LQR: Dict = {"source img": None, "prev img": None}
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    gamma_steps = int(stop_timestep * len(timesteps[:-1]))
    #gamma_steps = 9
    gamma = [0.9] * gamma_steps + [0] * (len(timesteps[:-1]) - gamma_steps)  # γ ∈ [0, 1] the controller guidance, γ can be time-varying

    if inverse:
        # todo if inverse, text prompt is φ
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        gamma = [0.5] * len(timesteps[:-1])  # γ ∈ [0, 1] the controller guidance

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    y1 = torch.randn(img.shape, device=img.device, dtype=img.dtype)
    
    y0, y_prev = None, None
    if img_LQR['source img'] is not None:
        y0 = img_LQR['source img'].to(img.device)
    if img_LQR['prev img'] is not None:
        y_prev = img_LQR['prev img'].to(img.device)

    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        pred, info = model(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )

        # 6. Unconditional Vector field uti(Yti) = u(Yti, ti, Φ(“”); φ)
        unconditional_vector_field = pred
        if not inverse:
            unconditional_vector_field = -unconditional_vector_field

        if inverse:
            # 7.Conditional Vector field  uti(Yti|y1) = (y1−Yti)/1−ti
            conditional_vector_field = (y1 - img) / (1 - t_curr)
        else:
            # 7.Conditional Vector field  uti(Xti|y0) = (y0−Xti)/(1−ti)
            t_i = i / len(timesteps[:-1])  # Empiracally better results
            #conditional_vector_field = (y0 - img) / t_curr
            if y_prev is None:
                conditional_vector_field = (y0 - img) / (1 - t_i)  
            else:
                #conditional_vector_field = (y_prev - img) / (1 - t_i)
                conditional_vector_field = (y0 - img) / (1 - t_i) + 0.7 * ((y_prev - img) / (1 - t_i) - (y0 - img) / (1 - t_i))
        
        # 8. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
        controlled_vector_field = unconditional_vector_field + gamma[i] * (conditional_vector_field - unconditional_vector_field)

        # 9. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
        delta_t = t_prev - t_curr
        if delta_t < 0:
            delta_t = t_curr - t_prev
        img = img + delta_t * controlled_vector_field

    return img, info


def denoise_multi_turn_consistent(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    inverse,
    info, 
    guidance: float = 4.0,
    #img_ori: Optional[Tensor] = None
    img_LQR: Dict = {"source img": None, "prev img": None}
):
    # this is ignored for schnell
    inject_list = [True] * info['inject_step'] + [False] * (len(timesteps[:-1]) - info['inject_step'])

    gamma_steps = int(info['lqr_stop'] * len(timesteps[:-1]))
    #gamma_steps = 9
    gamma = [0.9] * gamma_steps + [0] * (len(timesteps[:-1]) - gamma_steps)  # γ ∈ [0, 1] the controller guidance, γ can be time-varying

    if inverse:
        # todo if inverse, text prompt is φ
        timesteps = timesteps[::-1]
        inject_list = inject_list[::-1]
        gamma = [0.5] * len(timesteps[:-1])  # γ ∈ [0, 1] the controller guidance

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    step_list = []
    y1 = torch.randn(img.shape, device=img.device, dtype=img.dtype)

    y0, y_prev = None, None
    if img_LQR['source img'] is not None:
        y0 = img_LQR['source img'].to(img.device)
    if img_LQR['prev img'] is not None:
        y_prev = img_LQR['prev img'].to(img.device)

    next_step_velocity = None
    for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        info['t'] = t_prev if inverse else t_curr
        info['inverse'] = inverse
        info['second_order'] = False
        info['inject'] = inject_list[i]

        if next_step_velocity is None:
            pred, info = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
                info=info,
                cur_step=i
            )
        else:
            pred = next_step_velocity
        
        img_mid = img + (t_prev - t_curr) / 2 * pred

        t_vec_mid = torch.full((img.shape[0],), t_curr + (t_prev - t_curr) / 2, dtype=img.dtype, device=img.device)
        info['second_order'] = True
        pred_mid, info = model(
            img=img_mid,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec_mid,
            guidance=guidance_vec,
            info=info,
            cur_step=i
        )
        next_step_velocity = pred_mid

        # 6. Unconditional Vector field uti(Yti) = u(Yti, ti, Φ(“”); φ)
        unconditional_vector_field = pred_mid
        if not inverse:
            unconditional_vector_field = -unconditional_vector_field

        if inverse:
            # 7.Conditional Vector field  uti(Yti|y1) = (y1−Yti)/(1−ti)
            conditional_vector_field = (y1 - img) / (1 - t_curr + (t_prev - t_curr) / 2)
        else:
            # 7.Conditional Vector field  uti(Xti|y0) = (y0−Xti)/(1−ti)
            t_i = i / len(timesteps[:-1]) # Empiracally better results
            #conditional_vector_field = (y0 - img) / t_curr
            if y_prev is None:
                conditional_vector_field = (y0 - img) / (1 - t_i)  
            else:
                conditional_vector_field = (y0 - img) / (1 - t_i) + 0.7 * ((y_prev - img) / (1 - t_i) - (y0 - img) / (1 - t_i))
                #conditional_vector_field = (y_prev - img) / (1 - t_i)
        
        # 8. Controlled Vector field ti(Yti) = uti(Yti) + γ (uti(Yti|y1) − uti(Yti))
        controlled_vector_field = unconditional_vector_field + gamma[i] * (conditional_vector_field - unconditional_vector_field)

        # 9. Next state Yti+1 = Yti + ˆuti(Yti) (σ(ti+1) − σ(ti))
        delta_t = t_prev - t_curr
        if delta_t < 0:
            delta_t = t_curr - t_prev
        img = img + delta_t * controlled_vector_field

    return img, info
