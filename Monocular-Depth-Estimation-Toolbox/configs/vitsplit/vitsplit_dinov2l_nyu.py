_base_ = [
    '../_base_/default_runtime.py', 
    '../_base_/datasets/nyu.py', 
]
#########################
initial_lr = 3e-4
wd = 1e-2
lr_mult = 0.1
vit_hidden_dims = 1024
channels = 64
pergpu_batch_size = 4
total_layers = 24
frozen_num = 4
tuned_num = 3
start = 6
gap = (total_layers-1-start)/(frozen_num-1)
frozen_indices = [start+int(idx*gap) for idx in range(frozen_num)]
tuned_indices = [idx for idx in range(total_layers-tuned_num, total_layers)]

register_version = False
tune_register = False
dir_name = f"linear_dinov2l_vitsplit_nyu_{frozen_num}frozen_layers_{tuned_num}tuning_layers"
#########################

model = dict(
    type='DepthEncoderDecoder',
    backbone=dict(type="DINOViTSplitFusion",
                  backbone_size = "large",
                  register_version = register_version,
                  tune_register = tune_register,
                  out_indices=frozen_indices,
                  select_layers=tuned_indices,
                  channels=vit_hidden_dims,
                  tuning_type = "frozen"
                  ),
    decode_head=dict(
        type="LinearHead",
        in_channels=[vit_hidden_dims],
        channels=channels,
    ),
    test_cfg=dict(mode='whole')
)

optimizer = dict(
    type='AdamW',
    lr=initial_lr,
    betas=(0.9, 0.999),
    weight_decay=wd,
    paramwise_cfg=dict(
        custom_keys={
            'backbone.split_head': dict(lr_mult=lr_mult),
        }))

# learning policy
lr_config = dict(
    policy='OneCycle',
    max_lr=initial_lr,
    warmup_iters=1600 * 8,
    div_factor=25,
    final_div_factor=100,
    by_epoch=False)
# lr_config = dict(
#     policy="poly",
#     warmup="linear",
#     warmup_iters=1600*2,
#     warmup_ratio=1e-06,
#     power=0.9,
#     min_lr=0.0,
#     by_epoch=False,
# )
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# runtime settings
runner = dict(type='IterBasedRunner', max_iters=1600 * 24)
checkpoint_config = dict(by_epoch=False, max_keep_ckpts=2, interval=1600)
evaluation = dict(by_epoch=False, 
                  start=0,
                  interval=1600, 
                  pre_eval=True, 
                  rule='less', 
                  save_best='abs_rel',
                  greater_keys=("a1", "a2", "a3"), 
                  less_keys=("abs_rel", "rmse"))

# iter runtime
log_config = dict(
    _delete_=True,
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook') # TensorboardImageLoggerHook
    ])

data = dict(
    samples_per_gpu=pergpu_batch_size,
    workers_per_gpu=4,
)

find_unused_parameters=True
work_dir = f"./work_dirs/{dir_name}"