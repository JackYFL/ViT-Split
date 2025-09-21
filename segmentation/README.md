# Applying ViT-Split to Semantic Segmentation

Our segmentation code is developed on top of [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

For details see [ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads](https://arxiv.org/pdf/2506.03433).

If you find our work helpful, please star this repo and cite our paper:

```
@article{li2025vit,
  title={ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads},
  author={Li, Yifan and Li, Xin and Li, Tianqin and He, Wenbin and Kong, Yu and Ren, Liu},
  journal={arXiv preprint arXiv:2506.03433},
  year={2025}
}
```

## Installation

Install [MMSegmentation v0.20.2](https://github.com/open-mmlab/mmsegmentation/tree/v0.20.2).

```
# recommended environment: torch1.9 + cuda11.1
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.2 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install timm==0.4.12
pip install mmdet==2.22.0 # for Mask2Former
pip install mmsegmentation==0.20.2
ln -s ../detection/ops ./
cd ops & sh make.sh # compile deformable attention
```

or install everything directly

```bash
. install.sh
```

## Data Preparation

Preparing ADE20K/Cityscapes/Pascal Context according to the [guidelines](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) in MMSegmentation.

## Pretraining Sources
| Name          | Type       | Year | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---------- | ---- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DINOv2        | Self-Supervised        | 2023 | LVD-142M | [repo](https://github.com/facebookresearch/dinov2)                                            | [paper](https://arxiv.org/abs/2208.06366)     |


## Training and Evaluation
Train and evaluate our ViT-Split refer to the following script:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 . dist_train.sh configs/ade20k/vitsplit/linear_dinov2_vitsplit_small_512_20k_ade20k.py 4 --seed 2023 
```

Or using the bash script directly:

- **ADE20K**
```bash
# ViTSplit DINOV2-S 512*512
. scripts/ade20k/train_vitsplit_small.sh

# ViTSplit DINOV2-B 512*512
. scripts/ade20k/train_vitsplit_base.sh

# ViTSplit DINOV2-L 512*512
. scripts/ade20k/train_vitsplit_large.sh

# ViTSplit DINOV2-L 896*896
. scripts/ade20k/train_vitsplit_large_896.sh

# ViTSplit DINOV2-G 896*896
. scripts/ade20k/train_vitsplit_giant_896.sh
```

- **Cityscapes**
```bash
# ViTSplit DINOv2-S
. scripts/cityscapes/train_vitsplit_small.sh

# ViTSplit DINOv2-B
. scripts/cityscapes/train_vitsplit_base.sh

# ViTSplit DINOv2-L
. scripts/cityscapes/train_vitsplit_large.sh
```

- **Pascal Context**
```bash
# ViTSplit DINOv2-B
. scripts/pascal_context/train_vitsplit_base.sh

# ViTSplit DINOv2-L
. scripts/pascal_context/train_vitsplit_large.sh
```

## Results
The results are expected to be consistent with these, though minor variations may occur across different machines.

**ADE20K val**

| Model                 | Decoder | # Train Params | mIoU | Iterations |
|------------------------|---------|--------|------|------------|
| ViT-Split-S   | Linear  | 10.2M  | 51.6 | 40k        |
| ViT-Split-B   | Linear  | 40.5M  | 55.7 | 40k        |
| ViT-Split-L   | Linear  | 88.6M  | 58.2 | 40k        |
| ViT-Split-L (896)   | Linear  | 88.6M  | 59.0 | 40k        |
| ViT-Split-G (896)   | Linear  | 326M  | 60.2 | 40k        |

**Cityscapes val**
| Model                 | Decoder | # Train Params | mIoU | Iterations |
|------------------------|---------|--------|------|------------|
| ViT-Split-B   | Linear  | 55.2M  | 84.2 | 40k        |
| ViT-Split-L   | Linear  | 164.1M  | 85.8 | 40k        |

**Pascal Context**
| Model                 | Decoder | # Train Params | mIoU | Iterations |
|------------------------|---------|--------|------|------------|
| ViT-Split-B   | Linear  | 47.5M  | 66.4 | 20k        |
| ViT-Split-L   | Linear  | 164.1M  | 68.1 | 20k        |

## Acknowledgement

- Our codebase has referred to this wonderful codebase [ViT-Adapter](https://github.com/czczup/ViT-Adapter), thanks!
