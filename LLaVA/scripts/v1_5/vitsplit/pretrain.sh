#!/bin/bash
MODEL_NAME="vitsplit"

# GPU_ID="localhost:0"
GPU_ID="localhost:0,1,2,3,4,5,6,7"
SELECT_layers=(-18 -17 -16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1)
# FROZEN_SELECT_layers=(-3 -2)
FROZEN_SELECT_layers=(-2)
FROZEN_NUM=${#FROZEN_SELECT_layers[@]}
TUNED_NUM=1

deepspeed --include $GPU_ID llava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version plain \
    --data_path ./data/LLaVA-Pretrain/blip_laion_cc_sbu_558k.json \
    --image_folder ./data/LLaVA-Pretrain/ \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type vitsplit \
    --tune_mm_mlp_adapter True \
    --mm_frozen_num $FROZEN_NUM \
    --mm_frozen_select_layer ${FROZEN_SELECT_layers[@]} \
    --mm_tuned_layers_num $TUNED_NUM \
    --mm_vision_select_layer ${SELECT_layers[@]} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-$MODEL_NAME-pretrain-frozen$FROZEN_NUM-tuned$TUNED_NUM \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
