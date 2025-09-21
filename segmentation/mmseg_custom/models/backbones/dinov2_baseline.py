# Copyright (c) Bosch and ActionLab. All rights reserved.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from mmcv.runner import BaseModule
from mmseg.models.builder import BACKBONES

import torch.nn.functional as F
import torch.nn as nn
from functools import partial
import itertools
import math

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


# @BACKBONES.register_module()
# class DINOViT(BaseModule):
#     """DINO Vision Transformer."""

#     def __init__(
#         self,
#         backbone_size="small",
#         out_indices=[8,9,10,11],
#         tuning_type="frozen",
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         # load dinov2 from torch.hub
#         backbone_archs = {
#             'small': 'vits14',
#             'base': 'vitb14',
#             'large': 'vitl14',
#             'giant': 'vitg14',
#         }
#         # import ipdb; ipdb.set_trace()
#         backbone_arch = backbone_archs[backbone_size]
#         backbone_name = f'dinov2_{backbone_arch}'
#         torch.hub._validate_not_a_forked_repo=lambda a, b, c: True #modify a bug of torch 1.x for loading hub weights
#         backbone_model = torch.hub.load(
#             repo_or_dir = "facebookresearch/dinov2", model=backbone_name
#         )
#         # import ipdb; ipdb.set_trace()

#         # initialize backbone model
#         backbone_model.forward = partial(
#             backbone_model.get_intermediate_layers,
#             n = out_indices,
#             reshape = True,
#         )
#         backbone_model.register_forward_pre_hook(
#             lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
#         )

#         # Open the gradient of the given layer
#         if tuning_type == "frozen":
#             for param in backbone_model.parameters():
#                 param.requires_grad = False

#         elif tuning_type == "all":
#             for param in backbone_model.parameters():
#                 param.requires_grad = True

#         elif isinstance(tuning_type, list):
#             for param in backbone_model.parameters():
#                 param.requires_grad = False

#             for layer_id in tuning_type:
#                 for param in backbone_model.blocks[layer_id].parameters():
#                     param.requires_grad = True
#         else:
#             raise AttributeError(f"{tuning_type} is not supported !!!!")
#         self.backbone = backbone_model

#     def forward(self, x):
#         # import ipdb;ipdb.set_trace()
#         output_features = self.backbone(x)
#         return output_features


@BACKBONES.register_module()
class DINOViT(BaseModule):
    """DINO Vision Transformer."""

    def __init__(
            self,
            backbone_size="small",
            channels=384,
            out_indices=[8, 9, 10, 11],
            tuning_type="frozen",
            output_orgimg=False,
            patch_size=16,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.output_orgimg = output_orgimg
        # load dinov2 from torch.hub
        backbone_archs = {
            'small': 'vits14',
            'base': 'vitb14',
            'large': 'vitl14',
            'giant': 'vitg14',
        }
        # import ipdb; ipdb.set_trace()
        backbone_arch = backbone_archs[backbone_size]
        backbone_name = f'dinov2_{backbone_arch}'
        # modify a bug of torch 1.x for loading hub weights
        torch.hub._validate_not_a_forked_repo = lambda a, b, c: True
        backbone_model = torch.hub.load(
            repo_or_dir="facebookresearch/dinov2", model=backbone_name
        )
        # import ipdb; ipdb.set_trace()

        # initialize backbone model
        backbone_model.forward = partial(
            backbone_model.get_intermediate_layers,
            n=out_indices,
            reshape=True,
        )
        backbone_model.register_forward_pre_hook(
            lambda _, x: CenterPadding(backbone_model.patch_embed.patch_size[0])(x[0])
        )

        ###### resize patch size ######
        # import ipdb; ipdb.set_trace()
        # backbone_model = vit_resize_patch(backbone_model, patch_size=patch_size)
        ###############################

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

        # self.up1 = nn.Sequential(*[
        #     nn.ConvTranspose2d(channels, channels, 2, 2),
        #     nn.GroupNorm(32, channels),
        #     nn.GELU(),
        #     nn.ConvTranspose2d(channels, channels, 2, 2)
        # ])
        # self.up1 = nn.ConvTranspose2d(channels, channels, 2, 2)
        # self.up2 = nn.Identity()
        # self.up3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.up4 = nn.MaxPool2d(kernel_size=4, stride=4)

        # self.norm1 = nn.SyncBatchNorm(channels)
        # self.norm1 = nn.SyncBatchNorm(channels)
        # self.norm2 = nn.SyncBatchNorm(channels)
        # self.norm3 = nn.SyncBatchNorm(channels)
        # self.norm4 = nn.SyncBatchNorm(channels)

    def forward(self, x):
        # import ipdb;ipdb.set_trace()
        output_features = self.backbone(x)

        if len(output_features)==1: # for vitdet
            return output_features[0]
        else:
            # f1, f2, f3 = output_features
            f1, f2, f3, f4 = output_features
        
        # f1 = self.norm1(f1)
        # f2 = self.norm2(f2)
        # f3 = self.norm3(f3)
        # f4 = self.norm4(f4)

        # f1 = self.up1(f1)
        # f2 = self.up2(f2)
        # f3 = self.up3(f3)
        # f4 = self.up4(f4)

        return [f1, f2, f3, f4]
        # return [f1, f2, f3, f4]