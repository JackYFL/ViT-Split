from inspect import CO_VARARGS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from torch.nn.functional import embedding
from torch.nn.modules import conv

from depth.models.builder import HEADS
from .decode_head import DepthBaseDecodeHead
import torch.nn.functional as F
from depth.models.utils import UpConvBlock, BasicConvBlock

@HEADS.register_module()
class LinearHead(DepthBaseDecodeHead):
    """LinearHead.
    This head is implemented of `DenseDepth: <https://arxiv.org/abs/1812.11941>`_.
    Args:
        up_sample_channels (List): Out channels of decoder layers.
    """

    def __init__(self,
                 **kwargs):
        super(LinearHead, self).__init__(**kwargs)
        in_channels = self.in_channels[-1]
        self.up = nn.Sequential(*[
            nn.ConvTranspose2d(in_channels, in_channels//2, 2, 2),
            nn.GroupNorm(32, in_channels//2),
            nn.GELU(),
            nn.ConvTranspose2d(in_channels//2, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
        ])
        self.bn = nn.SyncBatchNorm(self.channels)

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self.up(inputs) # upsampling the image tokens
        x = self.bn(x) # B, D, H, W
        return x

    def forward(self, inputs, img_metas):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.depth_pred(output)
        return output
