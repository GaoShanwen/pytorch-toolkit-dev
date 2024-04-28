#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: feat_extract.py
# function: extract the features of custom data (train/val).
######################################################
import logging
from contextlib import suppress
from functools import partial

import numpy as np
import torch
import torch.nn.parallel
import tqdm
from timm.data import resolve_data_config
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import load_checkpoint
from timm.utils import reparameterize_model, setup_default_logging

from local_lib.data import create_custom_dataset, create_custom_loader
from local_lib.models import create_custom_model, FeatExtractModel, MultiLabelModel # enable local model
from local_lib.utils import parse_args
from local_lib.utils.file_tools import init_feats_dir, merge_feat_files, save_feat

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

try:
    from functorch.compile import memory_efficient_fusion

    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, "compile")

_logger = logging.getLogger("validate")


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
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            assert args.amp_dtype == "float16"
            use_amp = "apex"
            _logger.info("Validating in mixed precision with NVIDIA APEX AMP.")
        else:
            assert has_native_amp, "Please update PyTorch to a version with native AMP (or use APEX)."
            assert args.amp_dtype in ("float16", "bfloat16")
            use_amp = "native"
            amp_dtype = torch.bfloat16 if args.amp_dtype == "bfloat16" else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            _logger.info("Validating in mixed precision with native PyTorch AMP.")
    else:
        _logger.info("Validating in float32. AMP not enabled.")

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_custom_model(
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
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.model_classes = model.num_classes

    if args.feat_extract_dim is not None: #feat_extract
        model = FeatExtractModel(model, args.model, args.feat_extract_dim)
    if args.multilabel:
        model = MultiLabelModel(model, args.multilabel)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)
    
    if "redution" in args.model:
        import torch.nn as nn
        if "mobilenetv3" in args.model:
            model.classifier = nn.Identity()  # 移除分类层
        elif "regnet" in args.model:
            model.head.fc = nn.Identity()  # 移除分类层
        else:
            raise f"not support {args.model} !"

    if args.multilabel:
        model = model.base_model # 只留特征层
    elif args.feat_extract_dim is not None:
        model.classifier = nn.Identity()
        model.out_layer = nn.Flatten(1)
    
    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

    data_config = resolve_data_config(vars(args), model=model, use_test_size=not args.use_train_size, verbose=True)
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == "apex":
        model = amp.initialize(model, opt_level="O1")

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    root_dir = args.data or args.data_dir
    dataset = create_custom_dataset(
        name=args.dataset,
        root=root_dir,
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
    _logger.info(f"Loaded {args.infer_mode} task cats:{len(dataset.reader.class_to_idx)}, imgs={len(dataset)}")

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_custom_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config["crop_mode"],
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
        transfrom_mode="custom",
    )
    pbar = tqdm.tqdm(total=len(loader))

    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config["input_size"])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        init_feats_dir(args.results_dir)
        for batch_idx, (input, target) in enumerate(loader):
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

    pbar.close()
    img_files = dataset.reader.samples
    class_to_idx = dataset.reader.class_to_idx
    cat_list = None
    if not args.cats_path or (np.array(sorted(list(class_to_idx.values()))) == np.arange(args.data_classes)).all():
        cat_list = np.array(list(map(int, class_to_idx.keys())))
    # _logger.info(f"cat_list are {cat_list}.")
    merge_feat_files(args.results_dir, args.infer_mode, img_files, cat_list)
    _logger.info(f"feat saved in {args.results_dir}-{args.infer_mode}.npz")


if __name__ == "__main__":
    setup_default_logging()
    args, _ = parse_args()
    extract(args)
