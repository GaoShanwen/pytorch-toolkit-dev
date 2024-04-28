#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: validate.py
# function: validate custom valset / trainset.
######################################################
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel
from timm.data import create_loader, resolve_data_config
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import is_model, list_models, load_checkpoint
from timm.utils import (
    AverageMeter,
    accuracy,
    check_batch_size_retry,
    decay_batch_step,
    natural_key,
    reparameterize_model,
    set_jit_fuser,
    setup_default_logging,
)

from local_lib.data import RealLabelsCustomData, create_custom_dataset
from local_lib.models import create_custom_model, FeatExtractModel, MultiLabelModel
from local_lib.utils import ClassAccuracyMap, parse_args

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


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained and not args.checkpoint
    args.prefetcher = not args.no_prefetcher and args.multilabel is None

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

    if args.fuser:
        set_jit_fuser(args.fuser)

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
        num_classes=args.num_classes,
        in_chans=in_chans,
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
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info("Model %s created, param count: %d" % (args.model, param_count))

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

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data or args.data_dir
    dataset = create_custom_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        is_training=args.infer_mode == "train",
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_classes=args.num_classes,
        num_choose=args.num_choose,
        cats_path=args.cats_path,
        pass_path=args.pass_path,
        multilabel=args.multilabel,
    )

    if args.valid_labels:
        with open(args.valid_labels, "r") as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    # if args.real_labels:
    #     real_labels = RealLabelsCustomData(dataset.filenames(basename=True), real_file=args.real_labels)
    if args.results_file:
        real_labels = RealLabelsCustomData(dataset.filenames(basename=False), reader=dataset.reader)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config["crop_pct"]
    loader = create_loader(
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
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    if args.multilabel:
        attributes = args.multilabel.get("attributes", None)
        acc1_attrs = {attr: AverageMeter() for attr in attributes}
    else:
        acc_map = ClassAccuracyMap(dataset.reader.class_to_idx, args.label_file)
    model.eval()
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config["input_size"])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

                if valid_labels is not None:
                    output = output[:, valid_labels]
                if args.multilabel:
                    loss = model.module.get_loss(criterion, output, target)
                else:
                    loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            if args.multilabel:
                acc1, acc5, acc1_for_attrs = model.module.get_accuracy(accuracy, output, target, topk=(1, 5))
                for attr in attributes:
                    acc1_attrs[attr].update(acc1_for_attrs[attr].item(), input.size(0))
            else:
                acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            # acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))
            if args.compute_by_cat and not args.multilabel:
                acc_map.update(output.detach(), target, topk=(1, 5))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                rate_avg = input.size(0) / batch_time.avg
                attrs_info = ""
                if args.multilabel:
                    attrs_info = "Acc@1_ATTRS: "
                    attrs_info += "  ".join([
                        f"{attr}: {acc1.val:>7.3f} ({acc1.avg:>7.3f})" for attr, acc1 in acc1_attrs.items()
                    ])
                    
                _logger.info(
                    f"Test: [{batch_idx:>4d}/{len(loader)}]  "
                    f"Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  "
                    f"Loss: {losses.val:>7.4f} ({losses.avg:>6.4f})  "
                    f"Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  "
                    f"Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f}) "
                    f"{attrs_info}"
                )

    attrs_info = ""
    if args.results_file:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4),
        top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4),
        top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config["input_size"][-1],
        crop_pct=crop_pct,
        interpolation=data_config["interpolation"],
    )
    if args.multilabel:
        results["attributes"] = attributes
        attrs_info = "Acc@1_ATTRS: "
        for attr, acc1 in acc1_attrs.items():
            results[f"top1_attr_{attr}"] = round(acc1.avg, 4)
            results[f"top1_attr_{attr}_err"] = round(100 - acc1.avg, 4)
            attrs_info += f"{attr}: {acc1.avg:>7.3f}"
    if args.compute_by_cat and not args.multilabel:
        acc_map.save_to_csv(args.infer_mode)
    _logger.info(
        f" * Acc@1 {results['top1']:.3f} ({results['top1_err']:.3f})"
        f" Acc@5 {results['top5']:.3f} ({results['top5_err']:.3f})"
        f" {attrs_info}"
    )

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = "Unknown"
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and "cuda" in args.device:
                torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f"Reducing batch size to {batch_size} for retry.")
    results["error"] = error_str
    _logger.error(f"{args.model} failed to validate ({error_str}).")
    return results


_NON_IN1K_FILTERS = ["*_in21k", "*_in22k", "*in12k", "*_dino", "*fcmae", "*seer"]


def main():
    setup_default_logging()
    args, _ = parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + "/*.pth.tar")
        checkpoints += glob.glob(args.checkpoint + "/*.pth")
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == "all":
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, "") for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                args.model,
                pretrained=True,
            )
            model_cfgs = [(n, "") for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info("Running bulk validation on these pretrained models: {}".format(", ".join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if "error" in r:
                    continue
                if args.checkpoint:
                    r["checkpoint"] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x["top1"], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f"--result\n{json.dumps(results, indent=4)}")


def write_results(results_file, results, format="csv"):
    with open(results_file, mode="w") as cf:
        if format == "json":
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()


if __name__ == "__main__":
    main()
