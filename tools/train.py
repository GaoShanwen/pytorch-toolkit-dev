#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2023.11.09
# filenaem: train.py
# function: train custom data use timm.
######################################################
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime, timedelta
from functools import partial

import torch
import torch.nn as nn
import torchvision.utils
from timm import utils
from timm.data import AugMixDataset, FastCollateMixup, Mixup, create_loader, resolve_data_config
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import BinaryCrossEntropy, JsdCrossEntropy, LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import load_checkpoint, model_parameters, resume_checkpoint, safe_model_name
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import ApexScaler, NativeScaler
from torch.nn.parallel import DistributedDataParallel as NativeDDP

from local_lib.data import create_custom_dataset, CustomRandAADataset
from local_lib.models import create_custom_model, FeatExtractModel, MultiLabelModel # for regster local model
from local_lib.models.checkpoint import load_custom_checkpoint, filter_inconsistent_channels
from local_lib.utils import TensorBoardWriter, parse_args

try:
    from apex import amp
    from apex.parallel import DistributedDataParallel as ApexDDP
    from apex.parallel import convert_syncbn_model

    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.amp, "autocast") is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    import wandb

    has_wandb = True
except ImportError:
    has_wandb = False

has_compile = hasattr(torch, "compile")
_logger = logging.getLogger("train")

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128" #总是报显存不足的问题，是因为碎片没完全释放


def main():
    utils.setup_default_logging()
    args, args_text = parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher and args.multilabel is None
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            "Training in distributed mode with multiple processes, 1 device per process."
            f"Process {args.rank}, total {args.world_size}, device {args.device}."
        )
    else:
        _logger.info(f"Training with a single process on 1 device ({args.device}).")
    assert args.rank >= 0

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_dtype = torch.float16
    if args.amp:
        if args.amp_impl == "apex":
            assert has_apex, "AMP impl specified as APEX but APEX is not installed."
            use_amp = "apex"
            assert args.amp_dtype == "float16"
        else:
            assert has_native_amp, "Please update PyTorch to a version with native AMP (or use APEX)."
            use_amp = "native"
            assert args.amp_dtype in ("float16", "bfloat16")
        if args.amp_dtype == "bfloat16":
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(file=args.pretrained_path)

    model = create_custom_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        **factory_kwargs,
        **args.model_kwargs,
    )

    if args.feat_extract_dim is not None: #feat_extract
        model = FeatExtractModel(model, args.model, args.feat_extract_dim)
    if args.multilabel:
        model = MultiLabelModel(model, args.multilabel)

    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, "num_classes"), "Model must have `num_classes` attr if not set on cmd line/config."
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    if utils.is_primary(args):
        _logger.info(
            f"Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}"
        )

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, "A split of 1 makes no sense"
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device)
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ""  # disable dist_bn when sync BN active
        assert not args.split_bn
        if has_apex and use_amp == "apex":
            # Apex SyncBN used with Apex AMP
            # WARNING this won't currently work with models using BatchNormAct2d
            model = convert_syncbn_model(model)
        else:
            model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                "Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using "
                "zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled."
            )

    if args.torchscript:
        assert not args.torchcompile
        assert not use_amp == "apex", "Cannot use APEX AMP with torchscripted model"
        assert not args.sync_bn, "Cannot use SyncBatchNorm with torchscripted model"
        model = torch.jit.script(model)

    if not args.lr:
        global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = "sqrt" if any([o in on for o in ("ada", "lamb")]) else "linear"
        if args.lr_base_scale == "sqrt":
            batch_ratio = batch_ratio**0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f"Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) "
                f"and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling."
            )

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if use_amp == "apex":
        assert device.type == "cuda"
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
        loss_scaler = ApexScaler()
        if utils.is_primary(args):
            _logger.info("Using NVIDIA APEX AMP. Training in mixed precision.")
    elif use_amp == "native":
        try:
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        except (AttributeError, TypeError):
            # fallback to CUDA only AMP for PyTorch < 1.10
            assert device.type == "cuda"
            amp_autocast = torch.cuda.amp.autocast
        if device.type == "cuda" and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler()
        if utils.is_primary(args):
            _logger.info("Using native Torch AMP. Training in mixed precision.")
    else:
        if utils.is_primary(args):
            _logger.info("AMP not enabled. Training in float32.")

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        if args.finetune:
            load_custom_checkpoint(model, args.resume, strict=False)
        else:
            resume_epoch = resume_checkpoint(
                model,
                args.resume,
                optimizer=None if args.no_resume_opt else optimizer,
                loss_scaler=None if args.no_resume_opt else loss_scaler,
                log_info=utils.is_primary(args),
            )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            filter_fn = filter_inconsistent_channels if args.finetune else None
            load_checkpoint(model_ema.module, args.resume, use_ema=True, strict=False, filter_fn=filter_fn)
        if args.torchcompile:
            model_ema = torch.compile(model_ema, backend=args.torchcompile)

    # setup distributed training
    if args.distributed:
        if has_apex and use_amp == "apex":
            # Apex DDP preferred unless native amp is activated
            if utils.is_primary(args):
                _logger.info("Using NVIDIA APEX DistributedDataParallel.")
            model = ApexDDP(model, delay_allreduce=True)
        else:
            if utils.is_primary(args):
                _logger.info("Using native Torch DistributedDataParallel.")
            model = NativeDDP(model, device_ids=[device], broadcast_buffers=not args.no_ddp_bb)
        # NOTE: EMA model does not need to be wrapped by DDP

    if args.torchcompile:
        # torch compile should be done after DDP
        assert has_compile, "A version of torch w/ torch.compile() is required for --compile, possibly a nightly."
        model = torch.compile(model, backend=args.torchcompile)

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    
    dataset_train = create_custom_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        num_classes=args.num_classes,
        cats_path=args.cats_path,
        multilabel=args.multilabel,
    )
    cat_nums = len(dataset_train.reader.class_to_idx) if not args.multilabel else args.multilabel["label_nums"]
    _logger.info(f"Loaded trainset: cats={cat_nums}, imgs={len(dataset_train)}")

    dataset_eval = create_custom_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        num_classes=args.num_classes,
        cats_path=args.cats_path,
        multilabel=args.multilabel,
    )
    cat_nums = len(dataset_train.reader.class_to_idx) if not args.multilabel else args.multilabel["label_nums"]
    _logger.info(f"Loaded valset: cats={cat_nums}, imgs={len(dataset_eval)}")

    # setup mixup / cutmix
    collate_fn = None
    mixup_fn = None
    mixup_active = args.mixup > 0 or args.cutmix > 0.0 or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes,
        )
        if args.prefetcher:
            assert not num_aug_splits  # collate conflict (need to support deinterleaving in collate mixup)
            collate_fn = FastCollateMixup(**mixup_args)
        else:
            mixup_fn = Mixup(**mixup_args)

    # wrap dataset in AugMix helper
    if not args.custom_aa and num_aug_splits > 1:
        dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

    # create data loaders w/ augmentation pipeiine
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config["interpolation"]
    if args.custom_aa:
        assert args.aa.startswith('rand')
        dataset_train = CustomRandAADataset(
            dataset_train, 
            input_size=data_config["input_size"], 
            auto_augment=args.aa,
            interpolation=train_interpolation,
            mean=data_config["mean"],
            convert_epoch=args.convert_epoch,
        )
    loader_train = create_loader(
        dataset_train,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=True,
        use_prefetcher=args.prefetcher,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        distributed=args.distributed,
        collate_fn=collate_fn,
        pin_memory=args.pin_mem,
        device=device,
        use_multi_epochs_loader=args.use_multi_epochs_loader,
        worker_seeding=args.worker_seeding,
    )

    eval_workers = args.workers
    if args.distributed and ("tfds" in args.dataset or "wds" in args.dataset):
        # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
        eval_workers = min(2, args.workers)
    loader_eval = create_loader(
        dataset_eval,
        input_size=data_config["input_size"],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        use_prefetcher=args.prefetcher,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=eval_workers,
        distributed=args.distributed,
        crop_pct=data_config["crop_pct"],
        pin_memory=args.pin_mem,
        device=device,
    )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(smoothing=args.smoothing, target_threshold=args.bce_target_thresh)
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    tb_writer = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = "-".join(
                [
                    datetime.now().strftime("%Y%m%d-%H%M%S"),
                    safe_model_name(args.model),
                    str(data_config["input_size"][-1]),
                ]
            )
        output_dir = utils.get_outdir(args.output if args.output else "./output/train", exp_name)
        decreasing = True if eval_metric == "loss" else False
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing,
            max_history=args.checkpoint_hist,
        )
        with open(os.path.join(output_dir, "args.yaml"), "w") as f:
            f.write(args_text)

        if args.tensorboard:
            tb_writer = TensorBoardWriter(args.tensorboard, exp_name)

    if utils.is_primary(args) and args.log_wandb:
        if has_wandb:
            wandb.init(project=args.experiment, config=args)
        else:
            _logger.warning(
                "You've requested to log metrics to wandb but package not found. "
                "Metrics not being logged to wandb, try `pip install wandb`"
            )

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        _logger.info(
            f'Scheduled epochs: {num_epochs}. LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.'
        )

    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, "set_epoch"):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, "set_epoch"):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                train_loss_fn,
                args,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
            )

            if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == "reduce")

            eval_metrics = validate(
                model,
                loader_eval,
                validate_loss_fn,
                args,
                amp_autocast=amp_autocast,
            )

            if model_ema is not None and not args.model_ema_force_cpu:
                if args.distributed and args.dist_bn in ("broadcast", "reduce"):
                    utils.distribute_bn(model_ema, args.world_size, args.dist_bn == "reduce")

                ema_eval_metrics = validate(
                    model_ema,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    log_suffix=" (EMA)",
                )
                eval_metrics = ema_eval_metrics
            if tb_writer is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                tb_writer.update(epoch, train_metrics, eval_metrics, lr=sum(lrs) / len(lrs))
            if output_dir is not None:
                lrs = [param_group["lr"] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, "summary.csv"),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if saver is not None:
                # save proper checkpoint with eval metric
                save_metric = eval_metrics[eval_metric]
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=save_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch + 1, eval_metrics[eval_metric])

    except KeyboardInterrupt:
        pass

    if best_metric is not None:
        _logger.info("*** Best metric: {0} (epoch {1})".format(best_metric, best_epoch))


def train_one_epoch(
    epoch,
    model,
    loader,
    optimizer,
    loss_fn,
    args,
    device=torch.device("cuda"),
    lr_scheduler=None,
    saver=None,
    output_dir=None,
    amp_autocast=suppress,
    loss_scaler=None,
    model_ema=None,
    mixup_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False
    if hasattr(torch.cuda, 'empty_cache'): torch.cuda.empty_cache()

    second_order = hasattr(optimizer, "is_second_order") and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input = input.to(device)
            if isinstance(target, dict):
                target = {k: v.to(device) for k, v in target.items()}
            else:
                target = target.to(device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                output = model(input)
                if args.multilabel:
                    loss = model.module.get_loss(loss_fn, output, target)
                else:
                    loss = loss_fn(output, target)
            if accum_steps > 1:
                loss /= accum_steps
            return loss

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head="agc" in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head="agc" in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if has_no_sync and not need_update:
            with model.no_sync():
                loss = _forward()
                _backward(loss)
        else:
            loss = _forward()
            _backward(loss)

        if not args.distributed:
            losses_m.update(loss.item() * accum_steps, input.size(0))
        update_sample_count += input.size(0)

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == "cuda":
            torch.cuda.synchronize()
        time_now = time.time()
        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0:
            lrl = [param_group["lr"] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)
            # remaining_seconds: Estimated Time of Arrival
            remaining_seconds = round(
                ((args.epochs - epoch) * updates_per_epoch - (update_idx + 1)) * update_time_m.avg
            )

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                losses_m.update(reduced_loss.item() * accum_steps, input.size(0))
                update_sample_count *= args.world_size

            if utils.is_primary(args):
                _logger.info(
                    f"Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} "
                    f"({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  "
                    f"Loss: {losses_m.val:#.3g} ({losses_m.avg:#.3g})  "
                    f"Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  "
                    f"({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  "
                    f"LR: {lr:.3e}  "
                    f"ETA: {timedelta(seconds=remaining_seconds)}  "
                    # f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, "train-batch-%d.jpg" % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if saver is not None and args.recovery_interval and ((update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for
    
    if hasattr(optimizer, "sync_lookahead"):
        optimizer.sync_lookahead()

    return OrderedDict([("loss", losses_m.avg)])


def validate(model, loader, loss_fn, args, device=torch.device("cuda"), amp_autocast=suppress, log_suffix=""):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    if args.multilabel:
        attributes = args.multilabel.get("attributes", None)
        acc1_attrs = {attr: utils.AverageMeter() for attr in attributes}

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.no_grad():
        for batch_idx, (input, target) in enumerate(loader):
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device)
                if isinstance(target, dict):
                    target = {k: v.to(device) for k, v in target.items()}
                else:
                    target = target.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0 : target.size(0) : reduce_factor]

                if args.multilabel:
                    loss_calculator = model.module.get_loss if isinstance(model, NativeDDP) else model.get_loss
                    loss = loss_calculator(loss_fn, output, target)
                else:
                    loss = loss_fn(output, target)
            if args.multilabel:
                acc_calculator = model.module.get_accuracy if isinstance(model, NativeDDP) else model.get_accuracy
                acc1, acc5, acc1_for_attrs = acc_calculator(utils.accuracy, output, target, topk=(1, 5))
                for attr in attributes:
                    acc1_attrs[attr].update(acc1_for_attrs[attr].item(), input.size(0))
            else:
                acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == "cuda":
                torch.cuda.synchronize()

            losses_m.update(reduced_loss.item(), input.size(0))
            top1_m.update(acc1.item(), input.size(0))
            top5_m.update(acc5.item(), input.size(0))

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = "Test" + log_suffix
                _logger.info(
                    f"{log_name}: [{batch_idx:>4d}/{last_idx}]  "
                    f"Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  "
                    f"Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  "
                    f"Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  "
                    f"Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})"
                )

    if utils.is_primary(args) and args.multilabel:
        for attr in attributes:
            _logger.info(f"{attr} top1 accuracy: {acc1_attrs[attr].avg:>7.3f};")
    metrics = OrderedDict([("loss", losses_m.avg), ("top1", top1_m.avg), ("top5", top5_m.avg)])
    if args.multilabel:
        for attr in attributes:
            metrics.update({attr: acc1_attrs[attr].avg})

    return metrics


if __name__ == "__main__":
    main()
