import argparse
import copy
import logging
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
import mmseg_custom   # noqa: F401,F403
import mmcv_custom   # noqa: F401,F403
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction
from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.utils import collect_env, get_root_logger
from mmseg.models import build_segmentor

logger = logging.getLogger()
current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())


def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return [total_num, trainable_num]

def parse_args():
    parser = argparse.ArgumentParser("Segmentation Evaluation", add_help=False)
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--work-dir", help="the dir to save logs and models")
    parser.add_argument(
        "--load-denoiser-from", help="the checkpoint file to load weights from"
    )
    parser.add_argument("--resume-from", help="the checkpoint file to resume from")
    parser.add_argument(
        "--backbone-type",
        default="vit_small_patch14_dinov2.lvd142m",
        help="timm model type",
    )
    parser.add_argument("--stride", type=int, default=14)
    parser.add_argument("--disable_pe", action="store_true", default=False)
    parser.add_argument("--denoiser_type", type=str, default="transformer")
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="whether not to evaluate the checkpoint during training",
    )
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        "--gpus",
        type=int,
        help="(Deprecated, please use --gpu-id) number of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        help="(Deprecated, please use --gpu-id) ids of gpus to use "
        "(only applicable to non-distributed training)",
    )
    group_gpus.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help="id of gpu to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument(
        "--diff_seed",
        action="store_true",
        help="Whether or not set different seeds for different ranks",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="whether to set deterministic options for CUDNN backend.",
    )
    parser.add_argument("--headtype", type=str, default="linear", help="random seed")
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        "not be supported in version v0.22.0. Override some settings in the "
        "used config, the key-value pair in xxx=yyy format will be merged "
        "into config file. If the value to be overwritten is a list, it "
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        "marks are necessary and that no white space is allowed.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--auto-resume",
        action="store_true",
        help="resume from the latest checkpoint automatically.",
    )
    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args


def main(args):
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set cudnn_benchmark
    if cfg.get("cudnn_benchmark", False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get("work_dir", None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join(
            "./work_dirs", osp.splitext(osp.basename(args.config))[0]
        )
    # if args.load_from is not None:
    #     cfg.load_from = args.load_from
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpus is not None:
        cfg.gpu_ids = range(1)
        warnings.warn(
            "`--gpus` is deprecated because we only support "
            "single GPU mode in non-distributed training. "
            "Use `gpus=1` now."
        )
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn(
            "`--gpu-ids` is deprecated, please use `--gpu-id`. "
            "Because we only support single GPU mode in "
            "non-distributed training. Use the first GPU "
            "in `gpu_ids` now."
        )
    if args.gpus is None and args.gpu_ids is None:
        cfg.gpu_ids = [args.gpu_id]

    cfg.auto_resume = args.auto_resume

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == "none":
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        # gpu_ids is used to calculate iter when resuming checkpoint
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    # dump config
    # cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = osp.join(cfg.work_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # set multi-process settings
    # setup_multi_processes(cfg)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = "\n".join([f"{k}: {v}" for k, v in env_info_dict.items()])
    dash_line = "-" * 60 + "\n"
    logger.info("Environment info:\n" + dash_line + env_info + "\n" + dash_line)
    meta["env_info"] = env_info

    # log some basic info
    logger.info(f"Distributed training: {distributed}")
    logger.info(f"Config:\n{cfg}")

    # set random seeds
    # cfg.device = get_device()
    # seed = init_random_seed(args.seed, device=cfg.device)
    seed = init_random_seed(args.seed)
    seed = seed + dist.get_rank() if args.diff_seed else seed
    logger.info(f"Set random seed to {seed}, " f"deterministic: {args.deterministic}")
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta["seed"] = seed
    meta["exp_name"] = osp.basename(args.config)


    if "ms" in args.config:
        cfg.data.test.pipeline[1]["img_ratios"] = cfg.data.test.pipeline[1][
            "img_ratios"
        ][:3]
        # more scales: slower but better results, in (1,2,3,4,5)
        logger.info("scales: " + str(cfg.data.test.pipeline[1]["img_ratios"]))

    # import ipdb; ipdb.set_trace()
    model = build_segmentor(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg')
    )
    # model.init_weights
    # import ipdb; ipdb.set_trace()
    ## Print the number of trainable parameters ##
    # trainable_num_elements = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    # total_num_elements = sum(p.numel() for p in model.parameters()) 
    total_num_elements, trainable_num_elements = get_parameter_number(model)
    trainable_num_elements_M = trainable_num_elements / 1e6
    total_num_elements_M = total_num_elements / 1e6
    logger.info(f"Total size of trainable params: {trainable_num_elements_M:.2f}M, total size of params: {total_num_elements_M:.2f}M.")
    ## Print the number of trainable parameters ##
    # import ipdb; ipdb.set_trace()
    # shape = cfg.data['train']['pipeline'][3]['crop_size']
    # imgs = torch.randn(1, 3, shape[0], shape[1]).cuda()
    # flops, params = profile(model, inputs=(imgs, ))
    # logger.info(f"FLOPS: {flops / 1e9:.2f} GFLOPS")
    # logger.info(f"Params: {params / 1e6:.2f} M")
    # train

    # SyncBN is not support for DP
    if not distributed:
        warnings.warn(
            "SyncBN is only supported with DDP. To be compatible with DP, "
            "we convert SyncBN to BN. Please use dist_train.sh which can "
            "avoid this error."
        )
        model = revert_sync_batchnorm(model)

    logger.info(model)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    model.CLASSES = datasets[0].CLASSES
    start_time = time.time()
    train_segmentor(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta,
    )
    
    end_time = time.time()
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    elapsed_time = end_time - start_time

    elapsed_hours, remainder = divmod(elapsed_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(remainder, 60)

    # Print the results
    logger.info(f"Elapsed time: {int(elapsed_hours)} hours, {int(elapsed_minutes)} minutes, {int(elapsed_seconds)} seconds")
    logger.info(f"Total size of trainable params: {trainable_num_elements_M:.2f}M, total size of params: {total_num_elements_M:.2f}M.")
    # array = np.array(image)[:, :, ::-1] # BGR
    # segmentation_logits = inference_segmentor(model, array)[0]
    # segmented_image = render_segmentation(segmentation_logits, HEAD_DATASET)


if __name__ == "__main__":
    args = parse_args()
    main(args)