from typing import Any
from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.utils import colorstr
from .dataset import MixedDataset


def build_mixed_dataset(
    cfg: IterableSimpleNamespace,
    img_path: str,
    batch: int,
    data: dict[str, Any],
    mode: str = "train",
    rect: bool = False,
    stride: int = 32,
    multi_modal: bool = False,
    fraction: float | None = None,
) -> MixedDataset:
    pad = 0.0 if mode == "train" else 0.5
    if fraction is None:
        fraction = cfg.fraction if mode == "train" else 1.0

    return MixedDataset(
        img_path=img_path,
        imgsz=cfg.imgsz,
        batch_size=batch,
        augment=mode == "train",
        hyp=cfg,
        rect=cfg.rect or rect,
        cache=cfg.cache or None,
        single_cls=cfg.single_cls or False,
        stride=int(stride),
        pad=pad,
        prefix=colorstr(f"{mode}: "),
        task=cfg.task,
        classes=cfg.classes,
        data=data,
        fraction=fraction,
    )