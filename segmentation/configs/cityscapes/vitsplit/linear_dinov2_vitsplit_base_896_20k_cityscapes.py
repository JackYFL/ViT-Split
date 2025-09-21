# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../../_base_/datasets/cityscapes_896x896.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_20k.py'
]

#########################
crop_size = (896, 896)
initial_lr = 2e-4
wd = 1e-2
lr_mult = 0.1
vit_hidden_dims = 768
num_classes = 19 
pergpu_batch_size = 1
frozen_indices = [7, 8, 9, 10, 11]
tuned_indices = [7, 8, 9, 10, 11]
register_version = False
tune_register = False
work_dir = "./work_dirs/linear_dinov2-base_vitsplit_cityscapes20k_tuned5layers_frozenlast5layers"
#########################

find_unused_parameters=True
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type="EncoderDecoder",
    backbone=dict(type="DINOViTSplitFusion",
                  backbone_size="base",
                  channels = vit_hidden_dims,
                  register_version = register_version,
                  tune_register = tune_register,
                  out_indices=frozen_indices,
                  select_layers=tuned_indices,
                  tuning_type = "frozen"
                  ),
    decode_head=dict(
        type="ConvUpsampleLinearBNHead",
        in_channels=vit_hidden_dims,
        num_classes=num_classes,
        channels=vit_hidden_dims,
        norm_cfg=dict(type="SyncBN", requires_grad=True),
        loss_decode=dict(type="CrossEntropyLoss", use_sigmoid=False, loss_weight=1.0),
    ),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=(512, 512))
)
# dataset settings
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=(2048, 1024), ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),
    dict(type='ToMask'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg'])
]

optimizer = dict(type="AdamW", lr=initial_lr, weight_decay=wd, 
                paramwise_cfg=dict(
                custom_keys={
                    'backbone.split_head': dict(lr_mult=lr_mult)  # Apply a learning rate multiplier of 0.1 to the backbone
                }),
                betas=(0.9, 0.999)) # 0.001 for frozen

lr_config = dict(
    policy="poly",
    warmup="linear",
    warmup_iters=1500,
    warmup_ratio=1e-06,
    power=0.9,
    min_lr=0.0,
    by_epoch=False,
)

data = dict(samples_per_gpu=pergpu_batch_size,
            train=dict(pipeline=train_pipeline))
runner = dict(type='IterBasedRunner')
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=1000, metric='mIoU', save_best='mIoU')
