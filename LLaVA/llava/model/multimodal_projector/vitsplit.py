import torch
import torch.nn as nn
import re
import math

# class DeformableConvNet(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
#         super(DeformableConvNet, self).__init__()
#         self.offsets = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, padding=padding)
#         self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
   
#     def forward(self, x):
#         offsets = self.offsets(x)
#         x = self.deform_conv(x, offsets)
#         return x

class ViTSplit(nn.Module):
    def __init__(self, in_channels=1024, out_channels=4096, tuned_layers_num=1, frozen_select_layer=[-2]):
        super(ViTSplit, self).__init__()
        """
        ViTSplit for LLaVA
        """
        #############################
        frozen_num = len(frozen_select_layer)
        last_layer_id = -1
        self.last_layer_id = last_layer_id
        self.tuned_layers_num = tuned_layers_num
        self.frozen_select_layer = frozen_select_layer
        self.frozen_num = frozen_num
        if tuned_layers_num>0:
            from transformers import CLIPVisionModel
            vit_tower = CLIPVisionModel.from_pretrained('openai/clip-vit-large-patch14-336')

            if self.last_layer_id == 0:
                self.tuned_head = vit_tower.vision_model.encoder.layers[(-tuned_layers_num+last_layer_id):]
            else:
                self.tuned_head = vit_tower.vision_model.encoder.layers[(-tuned_layers_num+last_layer_id):last_layer_id]
            
            if frozen_num==0:
                self.proj = nn.Sequential(*[
                    nn.GELU(),
                    nn.Linear(in_channels, out_channels)
                ])
        if (tuned_layers_num>0) and (frozen_num>0):
            self.fusion_conv = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels*(frozen_num+1), out_channels=out_channels, kernel_size=1, padding=0),
                nn.GELU(),
                nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                nn.GELU()
            ])
        #############################

        if (tuned_layers_num > 0) and (frozen_num>0):
            self.frozen_conv = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels*frozen_num, out_channels=in_channels*frozen_num, kernel_size=1, padding=0),
                nn.GELU(),
                nn.Conv2d(in_channels=in_channels*frozen_num, out_channels=in_channels*frozen_num, kernel_size=3, padding=1),
                nn.GELU()
            ])
        elif (tuned_layers_num == 0) and (frozen_num>0):
            self.frozen_conv = nn.Sequential(*[
                nn.Conv2d(in_channels=in_channels*frozen_num, out_channels=in_channels*frozen_num, kernel_size=3, padding=0),
                nn.GELU(),
                # nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=1, padding=1),
                nn.Conv2d(in_channels=in_channels*frozen_num, out_channels=out_channels, kernel_size=1, padding=1),
                nn.GELU()
            ])


    def reshape_vit_tokens(self, x):
        """
        reshape vit tokens from (b, L, D) to (b, D, h, w)
        """
        b, L, _ = x.shape
        h = int(math.sqrt(L))
        x_ = x.reshape(b, h, h, -1).permute(0, 3, 1, 2).contiguous()
        return x_

    def forward(self, x):
        """
        x: L (B, N, D)
        return: (B, N, D')
        """
        if self.frozen_num>0:
            x_ = []
            for frozen_id in self.frozen_select_layer:
                x_.append(x[frozen_id])
            frozen_features = torch.cat(x_, dim=2) 

            frozen_features = self.reshape_vit_tokens(frozen_features)
            frozen_features = self.frozen_conv(frozen_features)
            if self.tuned_layers_num == 0:
                x = frozen_features

        if self.tuned_layers_num>0:
            tuned_features = x[-(self.tuned_layers_num+1)+self.last_layer_id]
            for blk in self.tuned_head:
                tuned_features = blk(tuned_features, None, None)
                tuned_features = tuned_features[0]
            x = self.reshape_vit_tokens(tuned_features)

        if (self.tuned_layers_num>0) and (self.frozen_num>0):
            x = torch.cat([frozen_features, x], dim=1)
            x = self.fusion_conv(x)

        (b, D, h, w) = x.shape
        x = x.reshape(b, D, -1).permute(0, 2, 1).contiguous()
        if self.frozen_num == 0:
            x = self.proj(x)
        return x