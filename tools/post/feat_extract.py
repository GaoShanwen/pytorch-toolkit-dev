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
import numpy as np
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import resolve_data_config
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import load_checkpoint
from timm.utils import setup_default_logging, ParseKwargs, reparameterize_model

import sys
sys.path.append('./')

from local_lib.models import create_owner_model # enable local model
from local_lib.data import create_owner_dataset
from local_lib.data import create_owner_loader

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

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--infer-mode', default="val", type=str,
                    metavar='NAME', help='the dirs to inference.')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
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
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--data-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--model-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--num-choose', type=int, nargs='+', default=None,
                    help='Number choose in dataset, (start_index, end_index)')
parser.add_argument('--start-batch', type=int, default=0,
                    help='start batch')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
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
parser.add_argument('--cats-path', default='', type=str, metavar='PATH',
                   help='path of dataset categories (default: none, current dir)')
parser.add_argument('--pass-path', default='dataset/exp-data/zero_dataset/pass_cats.txt', type=str, metavar='PATH',
                   help='path of pass categories (default: none, current dir)')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
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
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')


def save_feat(feats, idx, save_dir="./output/features"):
    np.savez(f'{save_dir}/{idx:08d}.npz', feats=feats)#, gts=gts.numpy())


def init_feats_dir(save_dir="./output/features"):
    if os.path.exists(save_dir):
        for file_name in os.listdir(save_dir):
            if not file_name.endswith(".npz"):
                raise f"{file_name} is error in {save_dir}!"
            os.remove(os.path.join(save_dir, file_name))
        os.removedirs(save_dir)
    os.makedirs(save_dir)


def merge_feat_files(load_dir="./output/features", infer_mode='val', file_infos=None):
    files = sorted(os.listdir(load_dir))
    feats, gts = [], []
    for file_name in files:
        file_path = os.path.join(load_dir, file_name)
        data = np.load(file_path)
        feats.append(data["feats"])
    merge_feats = np.concatenate(feats)
    merge_fpaths, merge_gts = zip(*(file_infos))
    merge_gts = np.array(merge_gts)
    np.savez(f'{load_dir}-{infer_mode}.npz', feats=merge_feats, gts=merge_gts, fpaths=merge_fpaths)


def extract(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            _logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info('Validating in float32. AMP not enabled.')

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_owner_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.model_classes,
        in_chans=in_chans,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )

    if args.model_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.model_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    if 'redution' in args.model:
        import torch.nn as nn
        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity() # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity() # 移除分类层
        else:
            raise f"not support {args.model} !"

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_owner_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        is_training=args.infer_mode=="train",
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_classes=args.data_classes,
        num_choose=args.num_choose,
        cats_path=args.cats_path,
        pass_path=args.pass_path,
    )
    _logger.info(f"load image number={len(dataset)}")
    
    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_owner_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
        transfrom_mode="owner"
    )
    pbar = tqdm.tqdm(total=len(loader)-args.start_batch)
    
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        if not args.start_batch: init_feats_dir(args.results_dir)
        for batch_idx, (input, target) in enumerate(loader):
            if batch_idx<args.start_batch:
                continue
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            
            # compute output
            with amp_autocast():
                output = model(input)
            save_feat(output.cpu().numpy(), batch_idx, args.results_dir)
            pbar.update(1)

    del pbar
    img_files = dataset.reader.samples
    merge_feat_files(args.results_dir, args.infer_mode, img_files)
    _logger.info(f'feat saved in {args.results_dir}-{args.infer_mode}.npz')


def main():
    setup_default_logging()
    args = parser.parse_args()

    extract(args)


if __name__ == '__main__':
    main()
