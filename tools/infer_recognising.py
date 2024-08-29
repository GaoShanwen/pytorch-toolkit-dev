#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2023.11.09
# filenaem: infer_recognising.py
# function: run model to recognise imgs and save (train/val).
######################################################
import logging
import os
import json
from contextlib import suppress

import numpy as np
import torch
import torch.nn.parallel
import tqdm
from timm.models import load_checkpoint, create_model
from timm.utils import setup_default_logging
from timm.utils.misc import natural_key

from local_lib.data.dataset_factory import create_custom_dataset
from local_lib.data.loader import create_custom_loader
from local_lib.models import FeatExtractModel, MultiLabelModel
from local_lib.utils.set_parse import parse_args

_logger = logging.getLogger("validate")
torch.cuda.empty_cache()


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

    model = create_model(
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


    if args.feat_extract_dim is not None: #feat_extract
        model = FeatExtractModel(model, args.model, args.feat_extract_dim)
    if args.multilabel:
        model = MultiLabelModel(model, args.multilabel)
    
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

    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    start_idx = 1 if args.multilabel else 0
    if args.input_mode == "file":
        need_index = 1
        if args.multilabel:
            need_index += args.multilabel["attributes"].index(args.need_attr)
        with open(args.data_path, "r") as f:
            load_data = list(zip(*([line.strip().split(",") for line in f.readlines()[start_idx:]])))
        query_files, query_labels = load_data[0], load_data[need_index]
        query_files, query_labels = np.array(query_files), np.array(query_labels)

        if args.cats_path:
            with open(args.cats_path, "r") as f:
                class_list = sorted([line.strip() for line in f.readlines()], key=natural_key)
            keeps = np.isin(query_labels, class_list)
            query_files, query_labels = query_files[keeps], query_labels[keeps]
    else: # args.input_mode == "path" or "dir"
        query_files = np.array(
            [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
            if args.input_mode == "dir"
            else [args.data_path]
        )
        query_labels = np.array([None] * len(query_files))
    
    class_to_idx = {c: idx for idx, c in enumerate(class_list)} if args.cats_path else None
    images_and_targets = list(zip(*(query_files, query_labels)))
    _logger.info(f"Loaded {len(query_files)} imgs")
    dataset = create_custom_dataset("txt_data", images_and_targets, class_to_idx=class_to_idx, split="infer")
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

    predicts = np.zeros((len(query_files), args.num_classes)).astype(np.float64)
    pbar = tqdm.tqdm(total=len(loader))
    model.eval()
    batch_size = args.batch_size
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            pbar.update()
            if device.type == "cuda":
                input = input.to(device)

            with amp_autocast():
                output = model(input)
            pred = output[args.need_attr] if args.multilabel else output
            start_idx = batch_idx * batch_size
            predicts[start_idx : start_idx + pred.shape[0]] = pred.softmax(dim=1).cpu().numpy()
             
    pbar.close()
    print("save prediction in recognize_scores.npz.")
    np.savez("recognize_scores.npz", pscores=predicts, gts=query_labels, files=query_files)


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    model = load_model(args)
    run_infer(model, args)
