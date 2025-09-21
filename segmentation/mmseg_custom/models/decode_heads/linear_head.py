import torch.nn as nn
import torch
from mmseg.models.builder import HEADS
from mmseg.models.decode_heads.decode_head import BaseDecodeHead


@HEADS.register_module()
class MultiScaleConvUpsampleLinearBNHead(BaseDecodeHead):
    """Using multiscale concatenation and deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        # self.cat_linear = nn.Linear(self.in_channels*4, self.in_channels)
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channels=self.channels*4, out_channels=self.channels, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv2d(in_channels=self.channels, out_channels=self.channels, kernel_size=3, padding=1),
            nn.GELU()
        ])
        self.up = nn.Sequential(*[
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2)
        ])
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # import ipdb; ipdb.set_trace()
        # print("inputs", [i.shape for i in inputs])
        # x = self._transform_inputs(inputs) # b, D, H, W
        x = torch.cat(inputs, dim=1) # b, 4*D, H, W
        x = self.conv(x)  # b, D, H, W
        x = self.up(x) # upsampling the image tokens
        # print("x", x.shape)
        x = self.bn(x) # B, D, H, W
        # print("feats", feats.shape)
        return x

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features. (num_scale, b, D, H, W)
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat": # 
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
                print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output) # b, D, H, W
        return output


@HEADS.register_module()
class ConvUpsampleLinearBNHead(BaseDecodeHead):
    """Using deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.up = nn.Sequential(*[
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
        ])
        self.resize_factors = resize_factors

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

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output) # b, D, H, W
        return output


@HEADS.register_module()
class LinearBNHead(BaseDecodeHead):
    """Using deconvolution for img tokens with batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        if isinstance(inputs, list):
            x = inputs[-1]
        else:
            x = inputs
        x = self.bn(x) # B, D, H, W
        return x

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output) # b, D, H, W
        return output


@HEADS.register_module()
class DconvUpsamplingBNHead(BaseDecodeHead):
    """Using Dconv Upsampling img tokens and batchnorm."""

    def __init__(self, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.up = nn.Sequential(*[
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
            nn.ConvTranspose2d(self.channels, self.channels, 2, 2),
            nn.GroupNorm(32, self.channels),
            nn.GELU(),
        ])
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # import ipdb; ipdb.set_trace()
        # print("inputs", [i.shape for i in inputs])
        # x = self._transform_inputs(inputs) # b, D, H, W
        # import ipdb; ipdb.set_trace()
        x = self.up(inputs) # upsampling the image tokens
        # print("x", x.shape)
        x = self.bn(x) # B, D, H, W
        # print("feats", feats.shape)
        return x

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features. (num_scale, b, D, H, W)
        Returns:
            Tensor: The transformed inputs
        """
        if self.input_transform == "resize_concat": # 
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
                print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        # import ipdb; ipdb.set_trace()
        output = self._forward_feature(inputs) # b, D, H, W
        output = self.cls_seg(output) # b, D, H, W
        return output
