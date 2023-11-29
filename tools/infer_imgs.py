#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_extract.py
# function: extract the features of owner data (train/val).
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
import torch.nn.parallel

from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging, ParseKwargs

import sys
sys.path.append('./')

import local_lib.models # enable local model
from local_lib.data.loader import owner_transfrom
from tools.post.feat_extract import save_feat, init_feats_dir
from tools.post.feat_tools import load_data
from tools.visualize.vis_error import load_csv_file, create_index, run_vis2bigimgs

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--data-path', default="", type=str,
                    metavar='NAME', help='the dirs to inference.')
parser.add_argument('--input-mode', default="", type=str,
                    help='the way of get input (path, dir, file).')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-g', "--gallerys", type=str, default="output/feats/regnety_040-train-0.16.npz")
parser.add_argument('-l', "--label-file", type=str,
                    default='dataset/exp-data/zero_dataset/label_names.csv')
parser.add_argument('-c', "--cats-file", type=str, 
                    default='dataset/exp-data/removeredundancy/629_cats.txt')
parser.add_argument('--use-gpu', action='store_true', default=False)
parser.add_argument("--topk", type=int, default=9)
parser.add_argument('--save-root', type=str, default='output/vis/errors')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mode-function', default='feat_extract', type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--num-choose', type=int, nargs='+', default=None,
                    help='Number choose in dataset, (start_index, end_index)')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
parser.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
parser.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
parser.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')
parser.add_argument('--results-dir', default='', type=str, metavar='FILEDIR',
                    help='Output feature file for validation results (summary)')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')


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
        args.model, pretrained=args.pretrained, num_classes=args.num_classes,
        in_chans=in_chans, drop_rate=args.drop, drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block, global_pool=args.gp, scriptable=args.torchscript,
        **args.model_kwargs,
    )

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, False)
    if 'redution' in args.model:
        import torch.nn as nn
        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity() # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity() # 移除分类层
        else:
            raise f"not support {args.model} !"

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

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

    with open(args.cats_file, 'r') as f: class_list = [line.strip('\n')[1:] for line in f.readlines()]
    assert args.input_mode in ["path", "dir", "file"], "please set infer_mode to path, dir, or files"
    if args.input_mode == "file":
        with open(args.data_path, 'r') as f:
            query_files, query_labels = zip(*([line.strip('\n').split(', ') for line in f.readlines()]))
        query_labels = [class_list.index(q_label) for q_label in query_labels]
        query_labels = np.array(query_labels, dtype=int)
        # import pdb; pdb.set_trace()
    else:
        query_files = os.listdir(args.data_path) if args.input_mode=="dir" else [args.data_path,]
        query_labels = None
    query_files = np.array(query_files)

    data_trans = owner_transfrom()
    pbar = tqdm.tqdm(total=query_files.shape[0])
    model.eval()
    init_feats_dir(args.results_dir)
    with torch.no_grad():
        for batch_idx, input in enumerate(query_files):
            input = data_trans(Image.open(input).convert('RGB')).unsqueeze(0)
            if args.no_prefetcher:
                input = input.to(device)
            
            with amp_autocast():
                output = model(input)
            save_feat(output.cpu().numpy(), batch_idx, args.results_dir)
            pbar.update(1)
    pbar.close()

    # args.param = f'IVF{args.num_classes},Flat'
    # args.measure = faiss.METRIC_INNER_PRODUCT
    args.param, args.measure = 'Flat', faiss.METRIC_INNER_PRODUCT
    # 加载npz文件
    gallery_feature, gallery_labels, gallery_files = load_data(args.gallerys)
    faiss.normalize_L2(gallery_feature)
    query_feats = load_feats(args.results_dir)
    faiss.normalize_L2(query_feats)
    
    label_index = load_csv_file(args.label_file)
    cats = list(set(gallery_labels))
    label_map = {i: label_index[cat] for i, cat in enumerate(class_list) if i in cats}
    index = create_index(gallery_feature, use_gpu=args.use_gpu, param=args.param, measure=args.measure)
    _, I = index.search(query_feats, args.topk)

    p_labels = gallery_labels[I]
    query_labels = query_labels if query_labels is not None else p_labels[:, 0]
    run_vis2bigimgs(I, gallery_labels, gallery_files, query_labels, query_files, label_map, args.save_root)


if __name__ == '__main__':
    setup_default_logging()
    args = parser.parse_args()
    model = load_model(args)
    run_infer(model, args)
    # matrix = np.array([
    #     [1.0, 0.9, 0.2, 0.1], 
    #     [0.2, 1.0, 0.1, 0.9],
    #     [0.2, 0.9, 1.0, 0.1], 
    #     [0.9, 0.2, 0.1, 1.0]
    # ])

    # masks = []
    # for i in range(matrix.shape[0]-1):
    #     if i in masks:
    #         continue
    #     masks += np.where((matrix[i, :] >= 0.8)&(np.arange(matrix.shape[0])>i))[0].tolist()
    # masks = np.array(masks)
    # # masks = np.where(
    # #     (matrix > 0.8)&
    # #     (np.arange(matrix.shape[1])[:, np.newaxis] > np.arange(matrix.shape[0]))
    # # )[0]
    # cat_index = np.arange(matrix.shape[0])
    # keep = np.setdiff1d(cat_index, cat_index[masks])
    # print(masks, keep)
