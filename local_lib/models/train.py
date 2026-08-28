"""
RF-DETR mixed-data trainer for the rfdetr-dev branch.

Provides class ID remapping and per-epoch background-image mixing without any
YOLO dependencies.  Usage::

    python tools/train.py --data <dataset_dir> --epochs 100 --batch 4 \\
        --class_mapping 5:0,6:1,7:2 --mixed_alpha 0.06 \\
        --background_data data/background.txt
"""
from __future__ import annotations

from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    RFDETRSegXLarge, RFDETRSeg2XLarge,
)
from rfdetr.utilities.logger import get_logger
from rfdetr.datasets.yolo import build_roboflow_from_yolo
from rfdetr.datasets.coco import build_roboflow_from_coco
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule as _OrigDM
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr._namespace import _namespace_from_configs

from ..data.mixed_data import MixedRFDETRDataset, MixedRFDETRDataModule

logger = get_logger(__name__)


def _select_model(args):
    MODEL_MAP = {
        "nano":    (RFDETRNano,     RFDETRSegNano),
        "small":   (RFDETRSmall,    RFDETRSegSmall),
        "medium":  (RFDETRMedium,   RFDETRSegMedium),
        "large":   (RFDETRLarge,    RFDETRSegLarge),
        "xlarge":  (None,           RFDETRSegXLarge),
        "2xlarge": (None,           RFDETRSeg2XLarge),
    }
    model_size = args.model or "medium"
    task = str(args.name or "").split("/")[0] or "detect"
    is_segment = "seg" in task.lower()
    pair = MODEL_MAP.get(model_size, (RFDETRMedium, RFDETRSegMedium))
    cls = (pair[1] if is_segment else pair[0]) if pair[0] or pair[1] else RFDETRMedium
    if cls is None:
        cls = pair[1] if pair[1] else pair[0]
    kwargs = {}
    if args.pretrained:
        kwargs["pretrain_weights"] = args.pretrained
    if args.resume:
        kwargs["resume"] = args.resume
    kwargs["resolution"] = args.imgsz[0]
    return cls(trust_checkpoint=True, **kwargs)


def train(args) -> None:
    """Train RF-DETR.  Dispatch to mixed-data path if class_mapping / mixed_alpha are set."""
    # ── Build model ─────────────────────────────────────────────────
    model = _select_model(args)

    # ── Build train config ──────────────────────────────────────────
    train_config = model.get_train_config(
        dataset_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        output_dir=f"{args.project}/{args.name}",
        lr=args.lr,
        grad_accum_steps=args.grad_accum_steps,
        num_workers=args.workers,
        # resolution=args.imgsz,
        # device=args.device,
    )

    # ── Build datasets ──────────────────────────────────────────────
    mc = model.model_config
    logger.info("Model config: %s", mc)
    ns = _namespace_from_configs(mc, train_config)
    ds_train = build_roboflow_from_coco("train", ns, mc.resolution)
    ds_val   = build_roboflow_from_coco("val",   ns, mc.resolution)

    # Build class names from the dataset's label2cat mapping (the authoritative
    # source for label→COCO-category_id remapping). Do not attempt to invent or
    # merge a separate ``final_list`` variable; that was unused in practice and
    # complicated the flow. Fail loudly if we cannot derive class names.
    label2cat = None
    if getattr(ds_train, "label2cat", None):
        label2cat = ds_train.label2cat
        coco_src = getattr(ds_train, "coco", None)
    elif getattr(ds_val, "label2cat", None):
        label2cat = ds_val.label2cat
        coco_src = getattr(ds_val, "coco", None)
    else:
        raise AssertionError("Could not find label2cat mapping on train or val dataset; cannot derive class names.")

    # Build a list where index == label index and value == class name string.
    max_label = max(label2cat.keys())
    class_names: list[str] = [None] * (max_label + 1)
    for label_idx, cat_id in label2cat.items():
        name = None
        if coco_src is not None and hasattr(coco_src, "cats") and cat_id in coco_src.cats:
            name = coco_src.cats[cat_id]["name"]
        else:
            name = f"class_{label_idx}"
        class_names[int(label_idx)] = str(name)

    # Sanity: no None entries
    assert all(n is not None for n in class_names), "Built class_names contains empty entries"

    # Apply the resolved class names to train_config and model config
    assert hasattr(train_config, "class_names"), "train_config does not support class_names; cannot proceed"
    train_config.class_names = class_names
    if hasattr(mc, "num_classes"):
        mc.num_classes = len(class_names)
    logger.info("Resolved class_names from dataset label2cat: %s", class_names)

    # ── Mixed-data path ───────────────────────────────────────────────
    opts = getattr(args, "options", {}) or {}
    class_mapping = opts.get("class_mapping")
    mixed_alpha = opts.get("mixed_alpha")
    background_data = opts.get("background_data")

    if class_mapping or (mixed_alpha and mixed_alpha > 0):
        bg_files: list[str] = []
        if background_data:
            try:
                with open(background_data, encoding="utf-8") as f:
                    bg_files = [ln.strip() for ln in f if ln.strip()]
            except FileNotFoundError:
                logger.warning("background_data file '%s' not found — continuing without background images.", background_data)
                bg_files = []
            except Exception:
                logger.exception("Error reading background_data '%s' — continuing without background images.", background_data)
                bg_files = []

        mixed = MixedRFDETRDataset(
            base_dataset=ds_train,
            class_mapping=class_mapping,
            background_im_files=bg_files or None,
            alpha=float(mixed_alpha) if mixed_alpha and mixed_alpha > 0 else None,
        )
        module = RFDETRModelModule(mc, train_config)
        assert hasattr(train_config, "class_names"), (
            "train_config must expose class_names before building RFDETRModelModule."
        )
        resolved_names = list(train_config.class_names)
        module.model.class_names = resolved_names
        if getattr(module.model, "args", None) is not None:
            setattr(module.model.args, "class_names", resolved_names)
        assert list(module.model.class_names) == resolved_names, (
            "final_list did not propagate into module.model.class_names"
        )
        logger.info(
            "Applied effective class_names to model head: %s (num_classes=%d)", resolved_names, len(resolved_names),
        )

        # Hook: resample backgrounds each epoch.
        _orig = getattr(module, "on_train_epoch_start", None)
        def _on_train_epoch_start():
            if callable(_orig):
                _orig()
            if mixed.alpha and mixed.alpha > 0 and mixed.background_im_files:
                mixed.resample_background()
        module.on_train_epoch_start = _on_train_epoch_start

        base_dm = _OrigDM(mc, train_config)
        # Best-effort: if the base datamodule already built its val split we can
        # wrap it immediately so validation uses remapped labels. Otherwise
        # MixedRFDETRDataModule.setup will handle wrapping before DataLoader
        # construction.

        base_dm._dataset_val = MixedRFDETRDataset(ds_val, class_mapping, background_im_files=None, alpha=None)

        datamodule = MixedRFDETRDataModule(base_dm, mixed)
        datamodule.class_names = resolved_names
        if hasattr(base_dm, '_dataset_train'):
            base_dm._dataset_train = mixed
    else:
        # Plain RF-DETR path: use upstream model.train()
        model.train(
            dataset_dir=args.data,
            epochs=args.epochs,
            batch_size=args.batch,
            output_dir=f"{args.project}/{args.name}",
            lr=args.lr,
            grad_accum_steps=args.grad_accum_steps,
            num_workers=args.workers,
            device=args.device,
        )
        logger.info("Training complete.")
        return

    # ── PTL trainer ─────────────────────────────────────────────────
    accelerator, devices = model._resolve_trainer_device_kwargs(args.device)
    trainer_kwargs: dict = {"accelerator": accelerator}
    # RF-DETR's helper returns `devices` as a list[int] when an explicit CUDA index
    # was provided. PyTorch-Lightning / build_trainer expects an int or a string
    # (e.g. "0" or "0,1") — convert lists accordingly.
    if devices is not None:
        if isinstance(devices, list):
            # Use device *count* when passing to PTL Trainer and let CUDA_VISIBLE_DEVICES
            # control which physical GPUs are visible. This avoids PTL's strict index format
            # differences across versions.
            trainer_kwargs["devices"] = len(devices)
        else:
            trainer_kwargs["devices"] = devices
    trainer = build_trainer(train_config, mc, **trainer_kwargs)
    # Pass datamodule as a keyword argument — newer Lightning versions interpret the
    # second positional arg as train_dataloaders, which would cause a TypeError if a
    # DataModule is provided positionally.
    trainer.fit(module, datamodule=datamodule, ckpt_path=train_config.resume or None)

    # ── Sync weights back ───────────────────────────────────────────
    model.model.model = module.model
    model.model.args  = _namespace_from_configs(mc, train_config)
    model._has_been_trained = True
    if hasattr(model, "remove_optimized_model"):
        model.remove_optimized_model()

    cfg_names = getattr(train_config, "class_names", None)
    assert cfg_names is not None, "train_config.class_names is required after training; final_list must be applied."
    model.model.class_names = list(cfg_names)
    if getattr(model.model, "args", None) is not None:
        setattr(model.model.args, "class_names", list(cfg_names))
    assert list(model.model.class_names) == list(cfg_names), (
        "final_list did not survive back-sync into model.model.class_names"
    )

    logger.info("Training complete.")