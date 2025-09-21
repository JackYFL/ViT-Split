# Applying ViT-Split to Monocular-Depth-Estimation
Our segmentation code is developed on top of [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox).

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

Please refer to [get_started.md](docs/get_started.md#installation) for installation.

## Data Preparation

Please refer to [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation. **Only** download **NYU** dataset.

## Pretraining Sources
| Name          | Type       | Year | Data         | Repo                                                                                                    | Paper                                                                                                                                                                           |
| ------------- | ---------- | ---- | ------------ | ------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| DINOv2        | Self-Supervised        | 2023 | LVD-142M | [repo](https://github.com/facebookresearch/dinov2)                                            | [paper](https://arxiv.org/abs/2208.06366)     |

## Training and Evaluation
Train and evaluate our ViT-Split refer to the following script
```bash
CUDA_VISIBLE_DEVICES=0 . ./tools/new_dist_train.sh configs/vitsplit/vitsplit_dinov2s_nyu.py 1 
```
Or using our scripts directly:
```bash
# DINOv2-small
. scripts/NYU/train_vitsplit-small.sh
# DINOv2-base
. scripts/NYU/train_vitsplit-base.sh
# DINOv2-large
. scripts/NYU/train_vitsplit-large.sh
```

## Results
The results are expected to be consistent with these, though minor variations may occur across different machines.

**NYU**
| Architecture     | Head       | #Train Param (↓) | AbsRel (↓) | RMSE (↓) | log₁₀ (↓) | δ₁ (↑) | δ₂ (↑) | δ₃ (↑) |
|------------------|------------|------------------|------------|----------|-----------|--------|--------|--------|
| DINOV2-G   | DPT       | –            | 0.0907     | 0.279    | 0.0371    | 0.9497 | 0.996  | 0.9994 |
| **ViT-Split-S** | Linear         | 9.3M         | 0.0897     | 0.3358   | 0.039     | 0.9327 | 0.9908 | 0.9985 |
| **ViT-Split-B** | Linear         | 37.0M        | 0.0853     | 0.3019   | 0.0365    | 0.9412 | 0.9947 | 0.9991 |
| **ViT-Split-L** | Linear         | 65.5M        | **0.078**  | **0.2672** | **0.0327** | **0.9622** | **0.9967** | **0.9994** |

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgement

This repo benefits from awesome works of [MDE-toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), [mmsegmentation](https://github.com/open-mmlab/mmsegmentation), [Adabins](https://github.com/shariqfarooq123/AdaBins),
[BTS](https://github.com/cleinc/bts). Please also consider citing them.
