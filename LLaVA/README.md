# Applying ViT-Split to LLaVA

Our VQA code is developed on top of [LLaVA-1.5](https://github.com/haotian-liu/LLaVA).

For details see [ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](https://arxiv.org/pdf/2506.03433).

If you find our work helpful, please star this repo and cite our paper:

```bibtex
@article{li2025vit,
  title={ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads},
  author={Li, Yifan and Li, Xin and Li, Tianqin and He, Wenbin and Kong, Yu and Ren, Liu},
  journal={arXiv preprint arXiv:2506.03433},
  year={2025}
}
```

## Installation

Install all the required packages using the following commands:
```Shell
conda create -n vitsplit-llava python=3.10 -y
conda activate vitsplit-llava
pip install --upgrade pip
pip install -e .

pip install -e ".[train]"
pip install flash-attn==2.6.3
```

or install everything directly

```bash
. install.sh
```

## Data Preparation

**Pretraining**: Please download the 558K subset of the LAION-CC-SBU dataset with BLIP captions [here](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain).

**Instruction tuning** Please download the annotation of the final mixture instruction tuning data [llava_v1_5_mix665k.json](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_v1_5_mix665k.json), and download the images from constituting datasets:

- COCO: [train2017](http://images.cocodataset.org/zips/train2017.zip)
- GQA: [images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip)
- OCR-VQA: [download script](https://drive.google.com/drive/folders/1_GYPY5UkUy7HIcR0zq3ZCFgeZN7BAfm_?usp=sharing), **we save all files as `.jpg`**
- TextVQA: [train_val_images](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip)
- VisualGenome: [part1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip), [part2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip)

After downloading all of them, organize the data as follows in `./playground/data`,

```
├── coco
│   └── train2017
├── gqa
│   └── images
├── ocr_vqa
│   └── images
├── textvqa
│   └── train_images
└── vg
    ├── VG_100K
    └── VG_100K_2
```


## LLaVA Weights
Please check LLaVA [Model Zoo](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md) for all public LLaVA checkpoints, and the instructions of how to use the weights.


## Train

LLaVA training consists of two stages: (1) feature alignment stage: use our 558K subset of the LAION-CC-SBU dataset to connect a *frozen pretrained* vision encoder to a *frozen LLM*; (2) visual instruction tuning stage: use 150K GPT-generated multimodal instruction-following data, plus around 515K VQA data from academic-oriented tasks, to teach the model to follow multimodal instructions.


### Pretrain (feature alignment)
In this stage, only the ViT-Split project is trained:
```bash
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
```

Or using the bash script directly:
```bash
. scripts/v1_5/vitsplit/pretrain.sh
```

### Visual Instruction Tuning
In this stage both the ViT-Split projector and the LLM are trained on instruction following question-answer pairs:
```bash
#!/bin/bash
MODEL_NAME="vitsplit"
GPU_ID="localhost:0,1,2,3,4,5,6,7"
# GPU_ID="localhost:0"
SELECT_layers=(-18 -17 -16 -15 -14 -13 -12 -11 -10 -9 -8 -7 -6 -5 -4 -3 -2 -1)
# FROZEN_SELECT_layers=(-3 -2)
FROZEN_SELECT_layers=(-2)
FROZEN_NUM=${#FROZEN_SELECT_layers[@]}
TUNED_NUM=1

GPU_ID="localhost:0,1,2,3,4,5,6,7"

deepspeed --include $GPU_ID llava/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ./data/LLaVA-Instruct-150K/llava_v1_5_mix665k_new.json \
    --image_folder ./data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter ./checkpoints/llava-v1.5-7b-$MODEL_NAME-pretrain-frozen$FROZEN_NUM-tuned$TUNED_NUM/mm_projector.bin \
    --mm_projector_type vitsplit \
    --mm_frozen_num $FROZEN_NUM \
    --mm_frozen_select_layer ${FROZEN_SELECT_layers[@]} \
    --mm_tuned_layers_num $TUNED_NUM \
    --mm_vision_select_layer ${SELECT_layers[@]} \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.5-7b-$MODEL_NAME-frozen$FROZEN_NUM-tuned$TUNED_NUM \
    --num_train_epochs 1 \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
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
```

Or using the bash script directly:
```bash
. scripts/v1_5/vitsplit/finetune.sh
```

### Train two stages
You can directly train two stages using one script:
```bash
. scripts/v1_5/vitsplit/two_stages.sh
```

## Evaluation

You can evaluate our ViT-Split adapted LLaVA on these benchmarks:
```bash
MODEL_NAME="vitsplit"
FROZEN_NUM=1
TUNED_NUM=1
CKPT="llava-v1.5-7b-$MODEL_NAME-frozen$FROZEN_NUM-tuned$TUNED_NUM"

CUDA_VISIBLE_DEVICES=7 . scripts/v1_5/eval/mmbench.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/pope.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/sqa.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/mmvet.sh $CKPT 
CUDA_VISIBLE_DEVICES=2 . scripts/v1_5/eval/llavabench.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/vqav2.sh $CKPT 
CUDA_VISIBLE_DEVICES=0 . scripts/v1_5/eval/vizwiz.sh $CKPT 
CUDA_VISIBLE_DEVICES=1 . scripts/v1_5/eval/gqa.sh $CKPT 
```
Or using the script directly:
```bash
. scripts/v1_5/vitsplit/eval.sh
```

## Results
The results are expected to be consistent with these, though minor variations may occur across different machines.

| Method                 | LLM        | Image Size | Sample Size Pre | Sample Size Ft | VQAv2 | VizWiz | LLaVA-Wild | SciQA-IMG | MM-Vet | POPE rand | POPE pop | POPE adv | MMB  |
|-------------------------|------------|------------|-----------------|----------------|-------|--------|------------|-----------|--------|-----------|----------|----------|------|
| LLaVA-1.5          | Vicuna-7B  | 336²       | 558K            | 665K           | 78.5 | 50.0  | 65.4       | 66.8      | 31.1   | 87.3      | 86.2     | 84.2     | 64.3 |
| **LLaVA-1.5 + ViT-Split** | Vicuna-7B | 336²       | 558K            | 665K           | 78.2 | 51.7 | 71.1 | 70.4 | 31.2 | 88.5 | 87.4 | 86.1 | 66.4 |

## Acknowledgement

- [LLaVA](https://github.com/haotian-liu/LLaVA): the codebase we built upon, please also cite this wonderful paper if you use our codebase!
