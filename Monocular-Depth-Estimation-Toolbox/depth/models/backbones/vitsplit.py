# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2d
from depth.models.builder import BACKBONES
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial
import itertools
import math
import copy

###Hack torch to load DINOv2####
from torch import Tensor
import torch._tensor
try:
    torch._tensor._rebuild_from_type_v2
except AttributeError:
    def _rebuild_from_type_v2(func, new_type, args, state):
        ret = func(*args)
        if type(ret) is not new_type:
            ret = ret.as_subclass(new_type)
        # Tensor does define __setstate__ even though it doesn't define
        # __getstate__. So only use __setstate__ if it is NOT the one defined
        # on Tensor
        if (
            getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
            is not Tensor.__setstate__
        ):
            ret.__setstate__(state)
        else:
            ret = torch._utils._set_obj_state(ret, state)
        return ret

    torch._tensor._rebuild_from_type_v2 = _rebuild_from_type_v2

import torch._utils
try:
    torch._utils._set_obj_state
except AttributeError:
    def _set_obj_state(obj, state):
        if isinstance(state, tuple):
            if not len(state) == 2:
                raise RuntimeError(f"Invalid serialized state: {state}")
            dict_state = state[0]
            slots_state = state[1]
        else:
            dict_state = state
            slots_state = None

        # Starting with Python 3.11, the __dict__ attribute is lazily created
        # and is serialized as None when not needed.
        if dict_state:
            for k, v in dict_state.items():
                setattr(obj, k, v)
    
        if slots_state:
            for k, v in slots_state.items():
                setattr(obj, k, v)
        return obj

    torch._utils._set_obj_state = _set_obj_state
##Hack torch to load DINOv2####

class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    # @torch.inference_mode()
    def forward(self, x):
        pads = list(
            itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1])
        )
        output = F.pad(x, pads)
        return output


class DeformableConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(DeformableConvNet, self).__init__()
        self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
   
    def forward(self, x):
        offsets = self.offsets(x)
        x = self.deform_conv(x, offsets)
        return x


@BACKBONES.register_module()
class DINOViTSplitFusion(BaseModule):
    """DINO Vision Transformer Split version."""

    def __init__(
        self,
        backbone_size="small",
        register_version=False,
        tune_register=True,
        out_indices=[8,9,10,11],
        select_layers=[10,11],
        channels=384,
        tuning_type="frozen",
        output_orgimg=False,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.output_orgimg = output_orgimg
        self.register_version = register_version
        # load dinov2 from torch.hub
        if not register_version:
            backbone_archs = {
                'small': 'vits14',
                'base': 'vitb14',
                'large': 'vitl14',
                'giant': 'vitg14',
            }
        else:
            backbone_archs = {
                'small': 'vits14_reg',
                'base': 'vitb14_reg',
                'large': 'vitl14_reg',
                'giant': 'vitg14_reg',
            }     
        backbone_arch = backbone_archs[backbone_size]
        # import ipdb; ipdb.set_trace()
        backbone_name = f'dinov2_{backbone_arch}'
        torch.hub._validate_not_a_forked_repo=lambda a, b, c: True #modify a bug of torch 1.x for loading hub weights
        backbone_model = torch.hub.load(
            repo_or_dir = "facebookresearch/dinov2", model=backbone_name
        )
        if register_version:
            self.num_register_tokens = backbone_model.num_register_tokens

        # Selecting layers
        self.select_layers = select_layers
        split_head = []
        for layer_id in self.select_layers:
            copy_blk = copy.deepcopy(backbone_model.blocks[layer_id])
            split_head.append(copy_blk)
        self.split_head = nn.Sequential(*split_head)

        # Hook the forward features
        self.split_activations = None
        backbone_model.blocks[self.select_layers[0]-1].register_forward_hook(self.get_activation)

        # initialize backbone model
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n = out_indices,
            reshape = True,
        )
        self.patch_size = backbone_model.patch_embed.patch_size[0]
        backbone_model.register_forward_pre_hook(
            lambda _, x: CenterPadding(self.patch_size)(x[0])
        )
        # Open the gradient of the given layer
        if tuning_type == "frozen":
            for param in backbone_model.parameters():
                param.requires_grad = False

        elif tuning_type == "all":
            for param in backbone_model.parameters():
                param.requires_grad = True

        elif isinstance(tuning_type, list):
            for param in backbone_model.parameters():
                param.requires_grad = False

            for layer_id in tuning_type:
                for param in backbone_model.blocks[layer_id].parameters():
                    param.requires_grad = True
        else:
            raise AttributeError(f"{tuning_type} is not supported !!!!")
            
        if register_version and tune_register:
            # import ipdb; ipdb.set_trace()
            for param in backbone_model.register_tokens:
                param.requires_grad = True

        self.backbone = backbone_model

        self.frozen_conv = nn.Sequential(*[
            nn.Conv2d(in_channels=channels*len(out_indices), out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.fusion_conv = nn.Sequential(*[
            nn.Conv2d(in_channels=channels*2, out_channels=channels, kernel_size=1, padding=0),
            nn.GELU(),
            DeformableConvNet(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.GELU()
        ])
        # self.up = nn.Sequential(*[
        #     nn.ConvTranspose2d(channels, channels, 2, 2),
        #     nn.GroupNorm(32, channels),
        # ])
        self.bn_norm = nn.SyncBatchNorm(channels)

    @property
    def get_activation(self):
        def hook(model, input, output):
            self.split_activations = output.detach()
        return hook
    
    def reshape_vit_tokens(self, x, norm=True):
        """
        reshape vit tokens from (b, L, D) to (b, D, h, w), refering to dinov2/model/vision_transformers.py "get_intermediate_layers" function
        input: x(b, L, D)
        output: x_(b, D, h, w)
        """
        b, L, D = x.shape

        if norm:
            x = self.backbone.norm(x)
        x_ = x[:, 1:, :] # drop class tokens in the batch
        if self.register_version:
            x_ = x_[:, self.num_register_tokens:, :] # drop register tokens in the batch
        # resized_len = int(math.sqrt(L-1))
        x_ = x_.reshape(b, self.h, self.w, -1).permute(0, 3, 1, 2).contiguous()
        return x_

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        frozen_features = self.backbone(x)
        _, _, H, W = x.shape
        self.h, self.w = math.ceil(H/self.patch_size), math.ceil(W/self.patch_size)
        # split head features
        tuned_features = self.split_head(self.split_activations)
        tuned_features = self.reshape_vit_tokens(tuned_features)

        frozen_features = torch.cat(frozen_features, dim=1) # b,nD,h,w
        frozen_features = self.frozen_conv(frozen_features) # b,nD,h,w

        x = torch.cat([frozen_features, tuned_features], dim=1)
        x = self.fusion_conv(x)

        return x
