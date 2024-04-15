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
import json
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
from local_lib.utils.visualize import save_imgs, save_predictions
from local_lib.utils.file_tools import load_names
from timm.utils.misc import natural_key

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
    assert isinstance(args.threshold, float), f"threshold={args.threshold} is not supported!"
    assert args.status is not None, f"status={args.status} is not supported!"

    if args.cats_path:
        with open(args.cats_path, "r") as f:
            # class_list = sorted([line.strip() for line in f.readlines()[:args.num_classes]], key=natural_key)
            class_list = sorted([line.strip() for line in f.readlines()], key=natural_key)
    if args.need_cats:
        with open(args.need_cats, "r") as f:
            load_cats = json.load(f)
        assert len(load_cats) == 1, f"the longth of load_cats must be one, but get {len(load_cats)}"
        need_cats, related_cats = np.array(list(load_cats.keys())), np.squeeze(np.array(list(load_cats.values())))
        
        # need_cats = np.array(load_cats["original_categories"])
        # need_idx = np.array([class_list.index(line.strip()) for line in need_cats])
        # related_cats = np.array(load_cats["related_categories"])
        # related_idx = np.array([class_list.index(line.strip()) for line in related_cats])
        # with open(args.need_cats, "r") as f:
        #     need_list = np.array([class_list.index(line.strip()) for line in f.readlines()])
        #     need_cat = np.array(class_list)[need_list]
    else:
        need_cats, related_cats = None, None
    filter_cats = related_cats if args.status == "after_analyze" else need_cats
    checker_cats = need_cats if args.status == "after_analyze" else related_cats
    check_idx = np.array([class_list.index(line) for line in checker_cats]) if checker_cats else np.array([])
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, "r") as f:
            query_files, query_labels = zip(*[line.strip().split(",") for line in f.readlines()])
        query_files, query_labels = np.array(query_files), np.array(query_labels)
        if args.only_need:
            keeps = np.isin(query_labels, filter_cats)
            query_files, query_labels = query_files[keeps], query_labels[keeps]
    else: # args.input_mode == "path" or "dir"
        query_files = np.array(
            [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
            if args.input_mode == "dir"
            else [args.data_path]
        )
        query_labels = np.array([None] * len(query_files))
    
    class_to_idx = {c: idx for idx, c in enumerate(class_list)}
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

    choices, predicts = np.zeros(len(query_files)).astype(bool), []
    pbar = tqdm.tqdm(total=len(loader))
    model.eval()
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            pbar.update(1)
            if device.type == "cuda":
                input = input.to(device)

            with amp_autocast():
                output = model(input)
            if args.status == "before_analyze":
                _, pred = output.topk(1, 1, True, True)
                pred = pred.T.cpu()[0].numpy()
                # this_choices = (np.isin(pred, check_idx)).astype(bool)
                this_choices = (
                    np.isin(pred, check_idx) if check_idx.shape[0] else np.ones(pred.shape[0])
                ).astype(bool)
            elif args.status == "after_analyze":
                pred = output.softmax(dim=1).cpu().numpy()
                this_choices = (np.sum(pred[:, check_idx], axis=1) >= args.threshold).astype(bool)
                pred = check_idx[np.argmax(pred[:, check_idx], axis=1)]
            if not np.sum(this_choices):
                continue
            predicts += pred[this_choices].tolist()
            keeps = np.arange(pred.shape[0])[this_choices] + (batch_idx * args.batch_size)
            choices[keeps] = True
    pbar.close()
    predicts = np.array(predicts) #choices = np.array(choices)
    choices_files, choices_gts = query_files[choices], query_labels[choices]
    _logger.info(f"choices/all: {predicts.shape[0]}/{query_files.shape[0]}")
    label_maps = load_names(args.label_file, idx_column=0, name_column=-1, to_int=False)
    predicts = np.array(class_list)[predicts]
    if need_cats is not None:
        for cat in need_cats:
            choices_num = np.sum(np.isin(predicts, [cat]))
            _logger.info(f"cat={cat} name={label_maps[cat]} num: {choices_num}")

    # with open("after_removed.txt", "w") as f:
    #     choices_files, choices_gts = query_files[~choices], query_labels[~choices]
    #     for file_path, label in zip(choices_files, choices_gts):
    #         f.write(f"{file_path},{label}\n")
    # predicts = np.array([label_maps[l] for l in predicts])
    predicts = predicts.astype(str)
    choices_gts = np.array([label_maps[l] for l in choices_gts])
    # save_imgs(choices_files, predicts, args.results_dir)
    save_predictions(choices_files, predicts, choices_gts, args.results_dir)
    # np.savez(f"blacklist-{args.infer_mode}.npz", files=choices_files, labels=choices_type)


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    model = load_model(args)
    run_infer(model, args)
