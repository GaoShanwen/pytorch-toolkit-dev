#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: infer_recognising.py
# function: run model to recognise imgs and save (train/val).
######################################################
import logging
import os
import shutil
from contextlib import suppress

import numpy as np
import torch
import torch.nn.parallel
import tqdm
from timm.models import load_checkpoint
from timm.utils import setup_default_logging

from local_lib.data.dataset_factory import create_custom_dataset
from local_lib.data.loader import create_custom_loader
from local_lib.models import create_custom_model
from local_lib.utils.set_parse import parse_args
from local_lib.utils.visualize import save_imgs
from timm.utils.misc import natural_key
from tools.before.check_data import load_names

_logger = logging.getLogger("validate")


def load_model(args):
    # might as well try to validate something
    args.pretrained = args.pretrained and not args.checkpoint
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

    if args.cats_path and args.need_cats:
        with open(args.cats_path, "r") as f:
            class_list = sorted([line.strip() for line in f.readlines()[:args.num_classes]], key=natural_key)
        with open(args.need_cats, "r") as f:
            need_list = [class_list.index(line.strip()) for line in f.readlines()]
    else:
        need_list = None
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, "r") as f:
            query_files, query_labels = zip(*[line.strip().split(",") for line in f.readlines()])
            if args.only_need:
                need_cat = np.array(class_list)[need_list]
                query_files = [f for f, l in zip(query_files, query_labels) if l in need_cat]
    else: # args.input_mode == "path" or "dir"
        query_files = (
            [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
            if args.input_mode == "dir"
            else [args.data_path]
        )
    _logger.info(f"Loaded {len(query_files)} imgs")
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
            this_choices = (
                np.where(pred[:, np.newaxis] == need_list)[0] if False else np.arange(pred.shape[0])
                # np.where(pred[:, np.newaxis] == need_list)[0] if need_list is not None else np.arange(pred.shape[0])
            )
            choices_type += pred[this_choices].tolist()
            base_idx = batch_idx * args.batch_size
            choices += (this_choices + base_idx).tolist()
    pbar.close()
    choices_type = np.array(choices_type)
    choices = np.array(choices)
    query_files = np.array(query_files)
    choices_files = query_files[choices]
    print(f"choices/all: {choices_type.shape[0]}/{query_files.shape[0]}")
    if need_list is not None:
        for cat in need_list:
            choices_num = np.where(choices_type == cat)[0].shape[0]
            print(f"cat={cat} num: {choices_num}")

    # import pdb; pdb.set_trace()
    if args.only_need:
        choices_type = np.array(class_list)[choices_type]
        label_maps = load_names("dataset/zero_dataset/label_names.csv", idx_column=0, name_column=-1, to_int=False)
        # print(f"{label_maps[need_list[0]]}")
        choices_type = np.array([label_maps[l] for l in choices_type])
    save_imgs(choices_files, choices_type, args.results_dir)
    # np.savez(f"blacklist-{args.infer_mode}.npz", files=choices_files, labels=choices_type)


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    model = load_model(args)
    run_infer(model, args)
