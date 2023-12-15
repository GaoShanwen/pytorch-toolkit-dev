#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_extract.py
# function: run model to searching imgs and save (train/val).
######################################################
import argparse
import logging
import os
import tqdm
from PIL import Image
import numpy as np
from contextlib import suppress
import faiss
import torch
import torch.nn as nn
import torch.nn.parallel
from timm.models import load_checkpoint
from timm.utils import setup_default_logging

from local_lib.models import create_custom_model
from local_lib.data.loader import create_custom_loader
from local_lib.data.dataset_factory import create_custom_dataset
from local_lib.utils import parse_args

import sys

sys.path.append(".")

from tools.post.feat_extract import save_feat, init_feats_dir
from tools.post.feat_tools import load_data
from tools.visualize.vis_error import load_csv_file, create_index, run_vis2bigimgs


_logger = logging.getLogger("Extract feature")


def load_feats(load_dir="./output/features"):
    files = sorted(os.listdir(load_dir))
    feats = []
    for file_name in files:
        file_path = os.path.join(load_dir, file_name)
        data = np.load(file_path)
        feats.append(data["feats"])
    return np.concatenate(feats)


def load_model(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint

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
    if "redution" in args.model:
        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity()  # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity()  # 移除分类层
        else:
            raise f"not support {args.model} !"

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

    with open(args.cats_file, "r") as f:
        class_list = [line.strip("\n") for line in f.readlines()]
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, "r") as f:
            query_files, query_labels = zip(*([line.strip("\n").split(", ") for line in f.readlines()]))
        query_labels = [
            class_list.index(q_label) if q_label in class_list else int(q_label) for q_label in query_labels
        ]
        query_labels = np.array(query_labels, dtype=int)
    else:
        query_files = (
            [os.path.join(args.data_path, path) for path in os.listdir(args.data_path)]
            if args.input_mode == "dir"
            else [args.data_path]
        )
        query_labels = None

    # query_files = np.array(query_files)
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

    # data_trans = custom_transfrom()
    pbar = tqdm.tqdm(total=len(loader))
    model.eval()
    init_feats_dir(args.results_dir)
    with torch.no_grad():
        for batch_idx, (input, _) in enumerate(loader):
            # import pdb; pdb.set_trace()
            # input = data_trans(Image.open(input).convert("RGB")).unsqueeze(0)
            if device == "cuda":
                input = input.to(device)

            with amp_autocast():
                output = model(input)
            save_feat(output.cpu().numpy(), batch_idx, args.results_dir)
            pbar.update(1)
    pbar.close()

    args.param = f"IVF629,Flat"
    args.measure = faiss.METRIC_INNER_PRODUCT
    # args.param, args.measure = "Flat", faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    gallery_feature, gallery_labels, gallery_files = load_data(args.gallerys)
    faiss.normalize_L2(gallery_feature)
    query_feats = load_feats(args.results_dir)
    faiss.normalize_L2(query_feats)

    searched_data = np.load("output/feats/searched_res-148c.npy")
    choose_idx = np.unique(searched_data[:, 1:].reshape(-1), return_index=True)[0]
    # import pdb; pdb.set_trace()
    gallery_feature = gallery_feature[choose_idx]
    index = create_index(gallery_feature, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    _, I = index.search(query_feats, args.topk)

    # res = np.concatenate((query_labels[:, np.newaxis], I), axis=1)
    # np.save('output/feats/searched_res-148c.npy', res)

    search_res = {data[0]: data[1:].tolist() for data in searched_data}
    tp_nums = 0
    for q_l, g_idx in zip(query_labels, I):
        g_searhed = search_res[q_l]
        pred_tp = np.in1d(choose_idx[g_idx], g_searhed)
        # import pdb; pdb.set_trace()
        if pred_tp.any():
            tp_nums += 1
    print(tp_nums, query_labels.shape[0])

    # p_labels = gallery_labels[I]
    # cats = list(set(gallery_labels))
    # label_index = load_csv_file(args.label_file)
    # label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    # query_labels = query_labels if query_labels is not None else p_labels[:, 0]
    # run_vis2bigimgs(I, gallery_labels, gallery_files, query_labels, query_files, label_map, args.save_root)


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    model = load_model(args)
    run_infer(model, args)
