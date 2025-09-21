
# Copyright (c) Bosch and ActionLab. All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from mmcv.runner import BaseModule
from mmcv.ops import DeformConv2d
from mmseg.models.builder import BACKBONES
import torch.distributed as dist

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from functools import partial
import itertools
import math
import copy

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
class TimmViTSplitFusion(BaseModule):
    """Timm Vision Transformer Split version."""

    def __init__(
        self,
        VFM='SigLip-L',
        out_indices=[8,9,10,11],
        select_layers=[10,11],
        channels=384,
        tuning_type="frozen",
        *args,
        **kwargs,
    ):
        super().__init__()
        model_name_dict = {
            'MAE-B': "vit_base_patch16_224.mae",
            'MAE-L': "vit_large_patch16_224.mae",
            'CLIP-B': "vit_base_patch16_clip_224.dfn2b",
            'CLIP-L': "vit_large_patch14_clip_224.dfn2b",
            'DINOv2-B': "vit_base_patch14_dinov2",
            'DINOv2-L': "vit_large_patch14_dinov2",
            'EVA2-B': "eva02_base_patch14_224.mim_in22k",
            'EVA2-L': "eva02_large_patch14_224.mim_in22k",
            'SigLip-B': "vit_base_patch16_siglip_224.webli",
            'SigLip-L': "vit_large_patch16_siglip_256.webli",
            'SAM-B': "samvit_base_patch16.sa1b",
            'SAM-L': "samvit_large_patch16.sa1b"
        }
        model_name = model_name_dict[VFM]
        model_kwargs = {
            "model_name": encoder_name,
            "pretrained": True,
            "num_classes": 0,
        }
        import ipdb; ipdb.set_trace()
        backbone_model = timm.create_model(**model_kwargs)

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
        x_ = x_.reshape(b, self.h, self.w, -1).permute(0, 3, 1, 2).contiguous()
        return x_

    def forward(self, x):
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

