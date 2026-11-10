#!/usr/bin/env python3
######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.10
# filenaem: inference.py
# function: run a segmentation checkpoint on images and write the result as
#          a picture with the prediction and the label side by side.
#
# the pipeline:
#   1. the model is rebuilt from the checkpoint (the architecture stored in
#      the checkpoint wins over the one of the config).
#   2. the images are read and resized exactly like in the validation of the
#      training (the 'img_size' of the config) and inferred by batches (the
#      'batch_size' of the config).
#   3. the predicted mask is drawn on the original picture: the pixels of a
#      class are blended into it, e.g. the hair is drawn with 70% of the
#      colour of the class and 30% of the original pixels.
#   4. the result is written as a picture with two halves next to each
#      other: the label (or the original image when there is no label) on
#      the left and the prediction on the right.
#
# two modes:
#   1. a split of the dataset (the val split by default): every image of the
#      split is inferred and the result is written as
#      <label> | <prediction>.
#   2. a single image without a label: the result is written as
#      <image> | <prediction>.
#
# every setting is read from the 'inference' section of the yaml config, so
# the config of the training is used as it is.
######################################################
import os
import sys
import glob
import logging

# make the repo root importable when running as `python tools/inference.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# fix the ssl certificates / the huggingface endpoint *before* torch and
# huggingface_hub are imported
from local_lib.utils.net_setup import setup as setup_network

setup_network()

import cv2
import numpy as np
import torch

from local_lib.models import create_seg_model, get_preprocessing_params, load_checkpoint
from local_lib.data.seg_augmentation import build_augmentation
from local_lib.utils import parse_args
from local_lib.utils.visualize import (
    make_figure,
    parse_palette,
    save_image,
)

_logger = logging.getLogger("inference")

# the extensions of the files that are read as images
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


# ============================ the setup ============================
def setup_logging():
    """Configure the logging of the tool."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_device(cfg):
    """Return the device the inference runs on."""
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        _logger.warning("cuda is requested but not available, fallback to cpu.")
        device = "cpu"
    return torch.device(device)


def build_model_from_checkpoint(cfg, checkpoint_path, device, num_classes):
    """Rebuild the model, preferring the architecture stored in the checkpoint."""
    model_cfg = dict(cfg)
    if checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_cfg = payload.get("config") if isinstance(payload, dict) else None
        if saved_cfg:
            # the checkpoint knows its own architecture; keep the cli overrides
            for key in ("model", "encoder_name", "model_kwargs", "in_channels"):
                if key in saved_cfg and key not in cfg:
                    model_cfg[key] = saved_cfg[key]
            if not cfg.get("classes") and saved_cfg.get("classes"):
                model_cfg["classes"] = saved_cfg["classes"]

    # the checkpoint already holds the trained weights, so there is no need to
    # download the pretrained encoder again
    model_cfg["encoder_weights"] = None

    model = create_seg_model(model_cfg, num_classes=num_classes)
    model.to(device)
    if checkpoint_path:
        load_checkpoint(checkpoint_path, model, map_location=device, strict=False)
        _logger.info(f"loaded checkpoint: {checkpoint_path}")
    else:
        _logger.warning("no checkpoint given, the model has random weights.")
    model.eval()
    return model


def get_classes(cfg, checkpoint_path):
    """Return the names of the classes of the run."""
    classes = list(cfg.get("classes", []) or [])
    if not classes and checkpoint_path and os.path.exists(checkpoint_path):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved = payload.get("config", {}) if isinstance(payload, dict) else {}
        classes = list(saved.get("classes", []) or [])
    return classes


def resolve_checkpoint(cfg, args=None):
    """Return the checkpoint of the run.

    The ``--checkpoint`` of the command line wins over the
    ``inference.checkpoint`` of the config, which wins over the
    ``checkpoint`` of the config.
    """
    inf_cfg = cfg.get("inference", {}) or {}

    checkpoint_path = ""
    if args is not None:
        checkpoint_path = getattr(args, "checkpoint", "") or ""
    if not checkpoint_path:
        checkpoint_path = inf_cfg.get("checkpoint", "") or ""
    if not checkpoint_path:
        checkpoint_path = cfg.get("checkpoint", "") or ""

    if not checkpoint_path:
        raise ValueError(
            "a checkpoint is required, pass it with --checkpoint <path> or set "
            "'inference.checkpoint' in the config."
        )
    return checkpoint_path


# ============================ the inference ============================
def preprocess(images, mean, std, device):
    """Normalise a batch of images into the input tensor of the model.

    Args:
        images: the images, a ``uint8`` RGB array (or a batch of them).
        mean: the mean of the normalisation.
        std: the standard deviation of the normalisation.
        device: the device of the tensor.

    Returns:
        The tensor of the images.
    """
    tensor = images.astype(np.float32) / 255.0
    tensor = torch.from_numpy(tensor).permute(0, 3, 1, 2)  # NCHW
    # the tensor is moved to the device before the normalisation, because
    # 'mean' and 'std' live on the device too
    tensor = tensor.to(device, non_blocking=True)
    tensor = (tensor - mean.to(device)) / std.to(device)
    return tensor


@torch.no_grad()
def predict_batch(model, images, mean, std, device):
    """Run the model on a batch of images and return the predicted class ids.

    Args:
        model: the model.
        images: the images, a ``float32`` NCHW tensor already normalised.
        mean: the mean of the normalisation.
        std: the standard deviation of the normalisation.
        device: the device of the model.

    Returns:
        The predicted class ids, a ``(N, H, W)`` array.
    """
    logits = model(images)
    return logits.argmax(dim=1).cpu().numpy().astype(np.uint8)


def read_label(mask_path, num_classes):
    """Read a label mask and return it as an array of class ids."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise RuntimeError(f"failed to read the mask: {mask_path}")
    if num_classes:
        # the ids that are not a class are treated as the background
        mask = np.where(mask >= num_classes, 0, mask)
    return mask


# ============================ the sources ============================
def collect_images(path):
    """Collect the images of a file / a folder / a glob pattern."""
    path = os.path.expanduser(path)

    if os.path.isfile(path):
        return [path]

    if os.path.isdir(path):
        found = []
        for root, _dirs, files in os.walk(path):
            for name in sorted(files):
                if name.lower().endswith(IMG_EXTENSIONS):
                    found.append(os.path.join(root, name))
        return found

    # a glob pattern
    found = sorted(glob.glob(path))
    return [f for f in found if f.lower().endswith(IMG_EXTENSIONS)]


def collect_split(cfg, split=None):
    """Return the images and the labels of a split of the dataset.

    The mask path of an image is derived the same way as in the dataset: the
    'images' folder of the path is replaced by the 'labels' one.

    Returns:
        A tuple ``(images, labels)``; the label of an image is ``None`` when
        the mask file is missing.
    """
    from local_lib.data.seg_factory import _resolve_list_file

    data_cfg = cfg.get("data", {}) or {}
    if split is None:
        split = data_cfg.get("val_split", "val")
    list_file = _resolve_list_file(cfg, split, data_cfg)

    images_dir_name = data_cfg.get("images_dir_name", "images")
    labels_dir_name = data_cfg.get("labels_dir_name", "labels")
    mask_ext = data_cfg.get("mask_ext", ".png")

    images = []
    labels = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            img_path = os.path.expanduser(line)
            if not os.path.isabs(img_path):
                img_path = os.path.join(os.path.dirname(os.path.abspath(list_file)), img_path)

            # the mask path is derived the same way as in the dataset
            mask_path = img_path.replace(images_dir_name, labels_dir_name)
            mask_path = os.path.splitext(mask_path)[0] + mask_ext
            if not os.path.exists(mask_path):
                mask_path = None

            images.append(img_path)
            labels.append(mask_path)

    return images, labels


def collect_source(cfg):
    """Return the images and the labels of the source of the config.

    The source is picked in this order:
      1. ``inference.image``  -> a single image without a label;
      2. ``inference.source`` -> a folder or a glob of images;
      3. the split of the dataset (``inference.split``, the val split by
         default) -> every image with its label.

    Returns:
        A tuple ``(images, labels)``.
    """
    inf_cfg = cfg.get("inference", {}) or {}

    image = inf_cfg.get("image", "")
    if image:
        if not os.path.isfile(image):
            raise FileNotFoundError(f"the image does not exist: {image}")
        return [image], [None]

    source = inf_cfg.get("source", "")
    if source:
        images = collect_images(source)
        return images, [None] * len(images)

    images, labels = collect_split(cfg, inf_cfg.get("split", "") or None)
    return images, labels


# ============================ the settings ============================
def resolve_settings(cfg):
    """Read the settings of the inference from the config.

    Returns:
        A dict with the settings.
    """
    inf_cfg = cfg.get("inference", {}) or {}

    # the share of the colour of the mask: 0.7 -> 70% of the colour and 30%
    # of the original pixels
    ratio = float(inf_cfg.get("ratio", 0.7))

    out_dir = inf_cfg.get("out_dir", "output/vis")

    # the size the image is resized to before the inference.  it is the size
    # of the training ('img_size'), so the model sees the same kind of input
    # as in the validation of the training.  empty keeps the original size.
    img_size = inf_cfg.get("img_size", "")
    if img_size in ("", None):
        # fall back to the size of the training
        img_size = cfg.get("img_size", "")
    if img_size not in ("", None):
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        img_size = tuple(img_size)
        # the size is rounded down to a multiple of 32, exactly like the
        # augmentation of the training does
        img_size = (max(32, (img_size[0] // 32) * 32), max(32, (img_size[1] // 32) * 32))
    else:
        img_size = None

    # the batch size and the number of workers are the ones of the training,
    # so the inference runs the same way as its validation
    batch_size = inf_cfg.get("batch_size", "")
    if batch_size in ("", None):
        batch_size = cfg.get("batch_size", 8)
    batch_size = int(batch_size)
    if batch_size < 1:
        raise ValueError(f"'inference.batch_size' must be at least 1, got {batch_size}.")

    workers = inf_cfg.get("workers", "")
    if workers in ("", None):
        workers = cfg.get("workers", 4)
    workers = int(workers)

    # the classes that are drawn; every class but the background by default
    class_ids = inf_cfg.get("classes", "")
    if isinstance(class_ids, str):
        class_ids = [int(c) for c in class_ids.split(",") if c.strip().isdigit()]
    elif isinstance(class_ids, int):
        class_ids = [class_ids]
    elif class_ids:
        class_ids = [int(c) for c in class_ids]
    else:
        class_ids = None
    if not class_ids:
        # an empty list would draw nothing; None draws every class but the
        # background
        class_ids = None

    # the BGR colour of every class
    palette = inf_cfg.get("palette", None)
    if palette:
        palette = parse_palette(palette)
    else:
        palette = None

    # the titles of the two halves of the picture
    labels = inf_cfg.get("labels", None)
    if labels and len(labels) >= 2:
        labels = [str(labels[0]), str(labels[1])]
    else:
        labels = None

    return {
        "ratio": ratio,
        "out_dir": out_dir,
        "img_size": img_size,
        "batch_size": batch_size,
        "workers": workers,
        "class_ids": class_ids,
        "palette": palette,
        "labels": labels,
        "gap": int(inf_cfg.get("gap", 8)),
        "suffix": str(inf_cfg.get("suffix", "")),
        "quality": int(inf_cfg.get("quality", 95)),
    }


def build_output_name(image_path, out_dir, suffix=""):
    """Build the path of the produced picture."""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    name = f"{stem}{suffix}.jpg"
    return os.path.join(out_dir, name)


# ============================ the pipeline ============================
class InferenceDataset(torch.utils.data.Dataset):
    """The images of the inference, read and resized like in the training.

    The images are resized to ``img_size`` with the same pipeline as the
    validation of the training, so the model sees the same kind of input.

    Args:
        samples: the ``(image_path, label_path)`` pairs.
        img_size: the size the images are resized to; None keeps the
            original size.
    """

    def __init__(self, samples, img_size=None):
        self.samples = list(samples)
        self.img_size = img_size
        self.augmentation = build_augmentation("test", img_size) if img_size else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label_path = self.samples[index]

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to read the image: {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.augmentation is not None:
            # the mask is not needed here, only the image is resized
            image = self.augmentation(image=image)["image"]

        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)  # CHW
        return image, index


def build_loader(cfg, samples, settings):
    """Build the dataloader of the inference.

    The batch size and the number of workers are the ones of the training,
    so the inference runs the same way as the validation of the training.
    """
    from local_lib.data import create_seg_loader

    dataset = InferenceDataset(samples, settings["img_size"])
    loader = create_seg_loader(
        dataset,
        batch_size=settings["batch_size"],
        num_workers=settings["workers"],
        shuffle=False,
        seed=int(cfg.get("seed", 42)),
    )
    return dataset, loader


def run_inference(cfg, checkpoint_path="", images=None, labels=None, out_dir=None,
                  desc="infer"):
    """Run the model on the images and write the pictures.

    Args:
        cfg: the config of the run.
        checkpoint_path: the checkpoint to load.
        images: the images to infer; the source of the config is used when
            they are not given.
        labels: the label of every image; None when there is no label.
        out_dir: the folder the pictures are written to; the one of the
            config is used when it is not given.
        desc: the name of the progress bar.

    Returns:
        The list of the written pictures.
    """
    settings = resolve_settings(cfg)
    if out_dir:
        settings["out_dir"] = out_dir

    if images is None:
        images, labels = collect_source(cfg)
    if labels is None:
        labels = [None] * len(images)

    if not images:
        raise RuntimeError("no image to infer.")

    device = get_device(cfg)
    classes = get_classes(cfg, checkpoint_path)
    if not classes:
        raise ValueError(
            "'classes' must be defined in the config or stored in the checkpoint."
        )

    model = build_model_from_checkpoint(cfg, checkpoint_path, device, len(classes))
    mean, std = get_preprocessing_params(cfg)
    mean = mean.to(device)
    std = std.to(device)

    os.makedirs(settings["out_dir"], exist_ok=True)

    samples = list(zip(images, labels))
    dataset, loader = build_loader(cfg, samples, settings)

    written = []
    try:
        from tqdm import tqdm
        iterator = tqdm(loader, desc=desc, leave=False)
    except ImportError:
        iterator = loader

    with torch.inference_mode():
        for batch, indices in iterator:
            images_batch = batch.to(device, non_blocking=True)
            images_batch = (images_batch - mean) / std

            logits = model(images_batch)
            preds = logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for position, index in enumerate(indices):
                index = int(index)
                image_path, label_path = samples[index]

                # the original picture is drawn, the prediction is resized
                # back to its size
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                if image is None:
                    _logger.warning(f"failed to read the image: {image_path}")
                    continue

                pred = preds[position]
                if pred.shape[:2] != image.shape[:2]:
                    pred = cv2.resize(
                        pred,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                label = None
                if label_path:
                    label = read_label(label_path, len(classes))

                figure = make_figure(image, pred, label, classes, settings)

                out_path = build_output_name(
                    image_path, settings["out_dir"], settings["suffix"]
                )
                save_image(out_path, figure, quality=settings["quality"])
                written.append(out_path)

    return written


def main(argv=None):
    """Run the inference from the command line.

    Args:
        argv: the arguments; ``sys.argv`` is used when it is None.
    """
    setup_logging()
    cfg, args, args_text = parse_args(
        "Run a segmentation checkpoint and write the predictions as pictures",
        argv=argv,
    )

    checkpoint_path = resolve_checkpoint(cfg, args)

    # the source of the pictures: a single image, a folder or a split
    images, labels = collect_source(cfg)
    _logger.info(f"inferring {len(images)} images")

    written = run_inference(
        cfg,
        checkpoint_path=checkpoint_path,
        images=images,
        labels=labels,
        desc="infer",
    )

    out_dir = (cfg.get("inference", {}) or {}).get("out_dir", "output/vis")
    _logger.info(f"{len(written)} pictures written to {out_dir}")
    return written


if __name__ == "__main__":
    main()
