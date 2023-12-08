#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: infer_recognising.py
# function: run model to recognise imgs and save (train/val).
######################################################
import argparse
import logging
import os
import shutil
import tqdm
from PIL import Image
import numpy as np
from contextlib import suppress
import torch
import torch.nn.parallel

from timm.models import load_checkpoint
from timm.utils import setup_default_logging, ParseKwargs
import sys

sys.path.append("./")

from local_lib.models import create_custom_model  # enable local model
from local_lib.data.loader import custom_transfrom, create_custom_loader
from local_lib.data.dataset_factory import create_custom_dataset

try:
    from apex import amp

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

_logger = logging.getLogger("validate")


parser = argparse.ArgumentParser(description="PyTorch ImageNet Validation")
parser.add_argument("--data-path", default="", type=str, metavar="NAME", help="the dirs to inference.")
parser.add_argument("--input-mode", default="", type=str, help="the way of get input (path, dir, file).")
parser.add_argument(
    "--model",
    "-m",
    metavar="NAME",
    default="dpn92",
    help="model architecture (default: dpn92)",
)
parser.add_argument("-c", "--cats-path", type=str, default="dataset/blacklist/save_cats.txt")
parser.add_argument("--need-cats", type=str, default="dataset/blacklist/need_cats.txt")
parser.add_argument("--save-root", type=str, default="output/vis/errors")
parser.add_argument(
    "--img-size",
    default=None,
    type=int,
    metavar="N",
    help="Input image dimension, uses model default if empty",
)
parser.add_argument(
    "--in-chans",
    type=int,
    default=None,
    metavar="N",
    help="Image input channels (default: None => 3)",
)
parser.add_argument(
    "--input-size",
    default=None,
    nargs=3,
    type=int,
    metavar="N N N",
    help="Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty",
)
parser.add_argument(
    "--use-train-size",
    action="store_true",
    default=False,
    help="force use of train input size, even when test size is specified in pretrained cfg",
)
parser.add_argument(
    "--crop-pct",
    default=None,
    type=float,
    metavar="N",
    help="Input image center crop pct",
)
parser.add_argument(
    "--crop-mode",
    default=None,
    type=str,
    metavar="N",
    help="Input image crop mode (squash, border, center). Model default if None.",
)
parser.add_argument(
    "--mode-function",
    default="feat_extract",
    type=str,
    metavar="N",
    help="Input image crop mode (squash, border, center). Model default if None.",
)
parser.add_argument(
    "--mean", type=float, nargs="+", default=None, metavar="MEAN", help="Override mean pixel value of dataset"
)
parser.add_argument(
    "--std",
    type=float,
    nargs="+",
    default=None,
    metavar="STD",
    help="Override std deviation of of dataset",
)
parser.add_argument(
    "--interpolation",
    default="",
    type=str,
    metavar="NAME",
    help="Image resize interpolation type (overrides model)",
)
parser.add_argument("--num-classes", type=int, default=None, help="Number classes in dataset")
parser.add_argument(
    "--num-choose",
    type=int,
    nargs="+",
    default=None,
    help="Number choose in dataset, (start_index, end_index)",
)
parser.add_argument(
    "--gp",
    default=None,
    type=str,
    metavar="POOL",
    help="Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.",
)
parser.add_argument(
    "--log-freq",
    default=10,
    type=int,
    metavar="N",
    help="batch logging frequency (default: 10)",
)
parser.add_argument(
    "--checkpoint",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
parser.add_argument("--pretrained", dest="pretrained", action="store_true", help="use pre-trained model")
parser.add_argument("--num-gpu", type=int, default=1, help="Number of GPUS to use")
parser.add_argument(
    "--batch-size",
    type=int,
    default=128,
    metavar="N",
    help="Input batch size for training (default: 128)",
)
parser.add_argument(
    "-j",
    "--workers",
    type=int,
    default=4,
    metavar="N",
    help="how many training processes to use (default: 4)",
)
parser.add_argument(
    "--no-prefetcher",
    action="store_true",
    default=False,
    help="disable fast prefetcher",
)
parser.add_argument(
    "--pin-mem",
    action="store_true",
    default=False,
    help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
)
parser.add_argument(
    "--channels-last",
    action="store_true",
    default=False,
    help="Use channels_last memory layout",
)
parser.add_argument("--device", default="cuda", type=str, help="Device (accelerator) to use.")
parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument(
    "--torchscript",
    default=False,
    action="store_true",
    help="torch.jit.script the full model",
)
scripting_group.add_argument(
    "--torchcompile",
    nargs="?",
    type=str,
    default=None,
    const="inductor",
    help="Enable compilation w/ specified backend (default: inductor).",
)
scripting_group.add_argument(
    "--aot-autograd",
    default=False,
    action="store_true",
    help="Enable AOT Autograd support.",
)

parser.add_argument("--drop", type=float, default=0.0, metavar="PCT", help="Dropout rate (default: 0.)")
parser.add_argument(
    "--drop-connect",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop connect rate, DEPRECATED, use drop-path (default: None)",
)
parser.add_argument(
    "--drop-path",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop path rate (default: None)",
)
parser.add_argument(
    "--drop-block",
    type=float,
    default=None,
    metavar="PCT",
    help="Drop block rate (default: None)",
)
parser.add_argument(
    "--results-dir",
    default="",
    type=str,
    metavar="FILEDIR",
    help="Output feature file for validation results (summary)",
)
parser.add_argument(
    "--retry",
    default=False,
    action="store_true",
    help="Enable batch size decay & retry for single model validation",
)


def save_imgs(choose_files, choices_type, save_root):
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    for file_path, cat in zip(choose_files, choices_type):
        save_dir = os.path.join(save_root, f"{cat}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        shutil.copy(file_path, save_dir)


def load_model(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_custom_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )

    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, False)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))
    return model


def run_infer(model, args):
    device = torch.device(args.device)
    # resolve AMP arguments based on PyTorch / Apex availability
    amp_autocast = suppress

    with open(args.cats_path, "r") as f:
        class_list = [line.strip("\n") for line in f.readlines()]
    with open(args.need_cats, "r") as f:
        need_list = [class_list.index(line.strip("\n")) for line in f.readlines()]
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, "r") as f:
            query_files = [line.strip("\n").split(", ")[0] for line in f.readlines()]
    else:
        query_files = (
            os.listdir(args.data_path)
            if args.input_mode == "dir"
            else [
                args.data_path,
            ]
        )
    dataset = create_custom_dataset(root=query_files, name="txt_data", split="infer")
    input_size = [3, args.img_size, args.img_size]
    loader = create_custom_loader(
        dataset,
        input_size=input_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        crop_pct=1.0,
        device=device,
        transfrom_mode="custom",
    )

    choices, choices_type = [], []
    pbar = tqdm.tqdm(total=len(loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            pbar.update(1)
            if device.type == "cuda":
                input = input.to(device)

            with amp_autocast():
                output = model(input)
            _, pred = output.topk(1, 1, True, True)
            pred = pred.T.cpu()[0].numpy()
            this_choices = np.where(pred[:, np.newaxis] == need_list)[0]
            choices_type += pred[this_choices].tolist()
            base_idx = batch_idx * args.batch_size
            choices += (this_choices + base_idx).tolist()
    pbar.close()
    choices_type = np.array(choices_type)
    choices = np.array(choices)
    query_files = np.array(query_files)
    choices_files = query_files[choices]
    print(f"choices/all: {choices_type.shape[0]}/{query_files.shape[0]}")
    for cat in need_list:
        choices_num = np.where(choices_type == cat)[0].shape[0]
        print(f"cat={cat} num: {choices_num}")

    # save_imgs(choices_files, choices_type, args.results_dir)
    base_name = os.path.basename(args.data_path).split(".")[0]
    np.savez(f"blacklist-{base_name}.npz", files=choices_files, labels=choices_type)


if __name__ == "__main__":
    setup_default_logging()
    args = parser.parse_args()
    model = load_model(args)
    run_infer(model, args)
