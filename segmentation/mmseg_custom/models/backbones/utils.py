import torch
import torch.nn.functional as nnf
import os
import numpy as np
import torch.nn.functional as F

def resample_patch_embed(input_tensor, target_patch_size):
    """
    Resample the input tensor to the target patch size.

    Parameters:
        input_tensor (torch.Tensor): Input image tensor of shape (B, C, H, W)
        target_patch_size (int): Target patch size for resampling

    Returns:
        torch.Tensor: Resampled tensor with new patch size
    """
    # Get the original height and width
    _, _, H, W = input_tensor.shape
    
    # Calculate the new dimensions based on the target patch size
    scale_factor = target_patch_size / min(H, W)
    new_size = (int(H * scale_factor), int(W * scale_factor))
    
    # Resize the input tensor
    resampled_tensor = F.interpolate(input_tensor, size=new_size, mode='bilinear', align_corners=False)
    
    return resampled_tensor

def vit_resize_patch(model, patch_size=16):
    model.patch_embed.patch_size = (patch_size, patch_size)
    model.patch_embed.proj.kernel_size = (patch_size, patch_size)
    model.patch_embed.proj.stride = (patch_size, patch_size)
    model.patch_embed.proj.weight = torch.nn.Parameter(
        resample_patch_embed(
            model.patch_embed.proj.weight,
            patch_size
        )
    )

    return model


def masked_select_features(features, mask):
    """
    Efficiently extracts features of shape (B, k, ...) based on a mask.
    
    Parameters:
        features: Input feature matrix with shape (B, L, ...)
        mask: Mask matrix with shape (B, L), where each batch has k True values
    
    Returns:
        Output feature matrix with shape (B, k, ...)
    """
    # Get the dimensions B and L, and infer the shape of trailing dimensions
    B, L = features.shape[:2]
    trailing_dims = features.shape[2:]  # Capture any additional dimensions after L
    
    # Expand the mask to match the dimensions of features (B, L) -> (B, L, 1, 1, ...)
    # Using *trailing_dims to expand for any trailing dimensions
    mask_expanded = mask.view(B, L, *([1] * len(trailing_dims))).expand(-1, -1, *trailing_dims)
    
    # Use masked_select to directly extract elements from features that meet the mask condition
    selected_features = torch.masked_select(features, mask_expanded)  # Output shape: (B * k * ...)
    
    # Calculate k, the number of True values in each batch (assumed to be consistent across batches)
    k = mask.sum(dim=1).max().item()
    
    # Reshape to (B, k, ...)
    selected_features = selected_features.view(B, k, *trailing_dims)
    
    return selected_features
    
def sample_gumbel(shape, eps=1e-20, device='cpu'):
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_top_k(scores, k):
    # add Gumbel noise
    gumbel_noise = sample_gumbel(scores.size(), device=scores.device)
    y = scores + gumbel_noise
    # obtain Top-k indices
    _, topk_indices = torch.topk(y, k)
    return topk_indices

def one_hot_with_grad(indices, num_classes):
    one_hot = torch.zeros((indices.size(0), num_classes), device=indices.device)
    one_hot.scatter_(1, indices.unsqueeze(1), 1)
    # using STE obtaining grad during backprop
    return one_hot - one_hot.detach() + one_hot.detach()


def align_cnn_vit_features(vit_features_bchw: torch.Tensor, cnn_features_bchw: torch.Tensor,
                           vit_patch_size: int = 14, vit_stride: int = 7,
                           cnn_stride: int = 8) -> torch.Tensor:
    """
    Assumptions:
    1. CNN layers are fully padded, thus the feature in the top left corner is centered at the [0, 0] pixel in the image.
    2. ViT patch embed layer has no padding, thus the feature in the top left corner is centered at [vit_patch / 2, vit_patch / 2].
    3. Feature and pixel positions are based on square pixels and refer to the center of the square
       (hence `align_corners=True` in grid_sample)
    :param vit_features_bchw: input ViT features (device and dtype will be set according th them)
    :param cnn_features_bchw: input CNN features to be aligned to ViT features
    :param vit_patch_size:
    :param vit_stride:
    :param cnn_stride:
    :return: CNN features sampled at ViT grid positions
    """
    with torch.no_grad():
        dtype = vit_features_bchw.dtype
        device = vit_features_bchw.device

        # number of features (ViT/CNN) we got
        v_sz = vit_features_bchw.shape[-2:]
        c_sz = cnn_features_bchw.shape[-2:]

        # pixel position of the bottom right feature
        c_br = [(s_ - 1) * cnn_stride for s_ in c_sz]

        # pixel locations of ViT features
        vit_x = torch.arange(v_sz[1], dtype=dtype, device=device) * vit_stride + vit_patch_size / 2.
        vit_y = torch.arange(v_sz[0], dtype=dtype, device=device) * vit_stride + vit_patch_size / 2.
        # map pixel locations to CNN feature locations in [-1, 1] scaled interval

        vit_grid_x, vit_grid_y = torch.meshgrid(-1. - (1. / c_br[1]) + (2. * vit_x / c_br[1]),
                                                -1 - (1. / c_br[0]) + (2. * vit_y / c_br[0]))
        grid = torch.stack([vit_grid_x, vit_grid_y], dim=-1)[None, ...].expand(vit_features_bchw.shape[0], -1, -1, -1)
    grid.requires_grad_(False)  # do not propagate gradients to the grid, only to the sampled features.
    aligned_cnn_features = nnf.grid_sample(cnn_features_bchw, grid=grid, mode='bilinear',
                                           padding_mode='border', align_corners=True)
    return aligned_cnn_features


'''
source_coords: N x 2
target_coords: N x 2
fg_mask: H x W
'''
def filter_bb_foreground_pairs(source_coords, target_coords, fg_mask, resw=854, resh=476):
    fg_mask_source = nnf.grid_sample(fg_mask[None, None, ...].float(), 2 * (source_coords[None, None, ...] / torch.tensor([resw, resh]).cuda()) - 1).squeeze()
    fg_mask_source = fg_mask_source > 0
    if len(fg_mask_source.shape) < 1:
        fg_mask_source = fg_mask_source.unsqueeze(0)
    return source_coords[fg_mask_source], target_coords[fg_mask_source], fg_mask_source


def get_last_ckpt_iter(folder_path):
    files = os.listdir(folder_path)
    file_paths = [os.path.join(folder_path, file) for file in files if os.path.isfile(os.path.join(folder_path, file))]
    
    current_iters = [-1]
    for file_path in file_paths:
        current_iters.append(int(file_path.split("_")[-1].split(".")[0]))
    return max(current_iters)


def load_pre_trained_model(pre_trained_sd, target_model):
    target_sd = {}
    for k, v in pre_trained_sd.items():
        target_sd[k] = v
    target_model.load_state_dict(target_sd)
    return target_model


def get_feature_cos_sims(fs, ft):
    assert fs.shape == ft.shape

    fs_n = fs.norm(dim=1) # b x h x w
    ft_n = ft.norm(dim=1) # b x h x w
    return torch.einsum("bchw,bchw->bhw", fs, ft) / (fs_n * ft_n)


def get_vit_feature_coords_from_mask(h, w, step=7, patch_size=14, device="cuda"):
    half_ps = patch_size // 2
    x = torch.arange(half_ps, w - half_ps + 1, step=step, device=device).float()
    y = torch.arange(half_ps, h - half_ps + 1, step=step, device=device).float()
    yy, xx = torch.meshgrid(y, x)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)
    grid = torch.stack([xx, yy], dim=-1)
    return grid


def fix_random_seeds(seed=31):
    """
    Fix random seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)