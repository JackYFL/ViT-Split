# Copyright (c) OpenMMLab. All rights reserved.
from .mask2former_head import Mask2FormerHead
from .maskformer_head import MaskFormerHead
from .linear_head import MultiScaleConvUpsampleLinearBNHead, ConvUpsampleLinearBNHead, LinearBNHead, DconvUpsamplingBNHead

__all__ = [
    'MultiScaleConvUpsampleLinearBNHead',
    'ConvUpsampleLinearBNHead',
    'LinearBNHead',
    'MaskFormerHead',
    'Mask2FormerHead',
    'DconvUpsamplingBNHead'
]
