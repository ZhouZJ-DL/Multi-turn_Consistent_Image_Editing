import torch
from einops import rearrange
from torch import Tensor
import math
from torchvision.utils import save_image
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as transforms


def adaptive_attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor, txt_shape: int, img_shape: int, cur_step:int, cur_block:int, info) -> Tensor:
    q, k = apply_rope(q, k, pe)

    #x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = scaled_dot_product_attention(q, k, v, txt_shape, img_shape, cur_step, cur_block, info)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def attention(q: Tensor, k: Tensor, v: Tensor, pe: Tensor) -> Tensor:
    q, k = apply_rope(q, k, pe)

    x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    x = rearrange(x, "B H L D -> B L (H D)")

    return x


def rope(pos: Tensor, dim: int, theta: int) -> Tensor:
    assert dim % 2 == 0
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / (theta**scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def apply_rope(xq: Tensor, xk: Tensor, freqs_cis: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    xq_out = freqs_cis[..., 0] * xq_[..., 0] + freqs_cis[..., 1] * xq_[..., 1]
    xk_out = freqs_cis[..., 0] * xk_[..., 0] + freqs_cis[..., 1] * xk_[..., 1]
    return xq_out.reshape(*xq.shape).type_as(xq), xk_out.reshape(*xk.shape).type_as(xk)


def auto_mask(load_list, mask_accumulator, thre, info, mask_num = 4):

    mask_list = []
    for img_path in load_list:
        load_mask_img = Image.open(img_path).convert('L')
        # Define the transformation
        transform = transforms.PILToTensor()
        mask_tensor = transform(load_mask_img)
        mask_tensor = mask_tensor.to(device=mask_accumulator.device, dtype=mask_accumulator.dtype)  # Set device and dtype
        mask_tensor /= 255.0
        mask_list.append(mask_tensor)  # Collect masks

    # Sort masks based on their activation levels
    mask_list.sort(key=lambda x: x.sum().item(), reverse=True)
    # Select the 5 medium activated masks
    num_masks = len(mask_list)
    if num_masks > mask_num:
        #selected_masks = mask_list[num_masks//2 - mask_num : num_masks//2]
        start_block = info['attn_guidance']
        end_block = info['attn_guidance'] + mask_num
        if end_block > num_masks - 1:
            selected_masks = mask_list[-mask_num: ]
        else:
            selected_masks = mask_list[start_block: end_block]
    else:
        selected_masks = mask_list

     # Accumulate the selected masks
    for mask in selected_masks:
        mask_accumulator += mask

    mask_tensor = (mask_accumulator / len(selected_masks)).to(dtype=mask_accumulator.dtype)  # Average the masks and convert back to original dtype
    mask_tensor[mask_tensor >= thre] = 1
    mask_tensor[mask_tensor < thre] = 0

    return mask_tensor


# Efficient implementation equivalent to the following:
def scaled_dot_product_attention(query, key, value, txt_shape, img_shape, cur_step, cur_block, info, 
        token_index=2, layer=range(19), attn_mask=None, dropout_p=0.0, coefficient=10, tau=0.5, thre=0.3, 
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).cuda()
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

    if not info['inverse']:
        # GENERATE MASK
        txt_img_cross = attn_weight[:, :, -img_shape:, :txt_shape]  # lower left part
        # each column maps to a token's heatmap
        token_heatmap = txt_img_cross[:, :, :, token_index]  # Shape: [1, 24, 1024]    
        token_heatmap = token_heatmap.mean(dim=1)[0]  # Shape: [1024]
        min_val, max_val = token_heatmap.min(), token_heatmap.max()
        norm_heatmap = (token_heatmap - min_val) / (max_val - min_val)

        mask_img = torch.sigmoid(coefficient*(norm_heatmap - 0.5))
        
        H = W = int(math.sqrt(mask_img.size(0)))
        mask_img = mask_img.reshape(H, W)

        save_path = f'heatmap/step_{cur_step}_layer_{cur_block}_token{token_index}.png'
        load_path = [f'heatmap/step_{cur_step-1}_layer_{i}_token{token_index}.png' for i in layer]        #save_image(mask_img.unsqueeze(0), save_path)
        save_image(mask_img.unsqueeze(0), save_path)

        mask_img[mask_img >= thre] = 1
        mask_img[mask_img < thre] = 0
        #save_image(mask_img.unsqueeze(0), save_path)

        mask_tensor = torch.zeros_like(mask_img)  # Set mask_tensor as a zero tensor
        if cur_step > 3:
            mask_accumulator = torch.zeros_like(mask_tensor.unsqueeze(0), dtype=mask_img.dtype)  # Accumulator for averaging masks
            mask_tensor = auto_mask(load_path, mask_accumulator, thre, info, mask_num=4)
            if cur_block == 1:
                save_image(mask_tensor, f'heatmap/average_heatmaps/step_{cur_step}_layer_{cur_block}_token{token_index}.png')


        if not torch.all(mask_tensor == 0):
            highlight_factor = 2.0  # Factor to increase weights in the masked area
            reduce_factor = 0.8  # Factor to decrease weights in the unmasked area

            mask_tensor = mask_tensor.reshape(1, H * W)
            mask_tensor = mask_tensor.unsqueeze(1).unsqueeze(-1)
            # Create a multiplier tensor: 2.0 where mask is active, 0.5 where mask is inactive.
            multiplier = torch.where(mask_tensor.bool(), torch.tensor(highlight_factor), torch.tensor(reduce_factor))
            attn_weight[:, :, -img_shape:, :15] *= multiplier

    return attn_weight @ value

'''
    if cur_step == 14 and (cur_block == 2 or cur_block == 7 or cur_block == 12):
        mask_img = torch.zeros_like(mask_img)
        for j in range(5):
            token_heatmap = txt_img_cross[:, :, :, j]
            token_heatmap = token_heatmap.mean(dim=1)[0]
            min_val, max_val = token_heatmap.min(), token_heatmap.max()
            norm_heatmap = (token_heatmap - min_val) / (max_val - min_val)

            mask_img = torch.sigmoid(coefficient*(norm_heatmap - 0.5)) 
            
            H = W = int(math.sqrt(mask_img.size(0)))
            mask_img = mask_img.reshape(H, W)
            save_path = f'/home/hfle/personalization/FireFlow-Fast-Inversion-of-Rectified-Flow-for-Image-Semantic-Editing/heatmap/step_{cur_step}_layer_{cur_block}_token{j}.png'
            save_image(mask_img.unsqueeze(0), save_path)'''
