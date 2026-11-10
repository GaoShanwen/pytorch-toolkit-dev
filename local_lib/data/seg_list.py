######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_list.py
# function: build the yolo-style train / val txt lists of a segmentation dataset.
######################################################
import os
import random


IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def collect_images(root, images_dir_name="images", mask_ext=".png", labels_dir_name="labels"):
    """Walk ``root`` and return the image paths that have a matching mask.

    The mask of an image is derived yolo style: the ``images`` folder of the
    path is replaced by ``labels`` and the extension by ``mask_ext``.
    """
    from .seg_dataset import img2label_path

    root = os.path.abspath(os.path.expanduser(root))
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset root not found: {root}")

    images = []
    for dirpath, _, filenames in os.walk(root):
        for name in sorted(filenames):
            if not name.lower().endswith(IMG_EXTENSIONS):
                continue
            img_path = os.path.join(dirpath, name)
            mask_path = img2label_path(
                img_path, images_dir_name, labels_dir_name, mask_ext
            )
            if os.path.exists(mask_path):
                images.append(img_path)
    return sorted(images)


def write_image_list(image_paths, list_file):
    """Write the image paths into the txt list, one path per line."""
    os.makedirs(os.path.dirname(os.path.abspath(list_file)), exist_ok=True)
    with open(list_file, "w", encoding="utf-8") as f:
        for path in image_paths:
            f.write(f"{os.path.abspath(path)}\n")
    return list_file


def create_dataset_lists(
    root,
    out_dir=None,
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    images_dir_name="images",
    labels_dir_name="labels",
    mask_ext=".png",
):
    """Scan ``root`` and write the ``train.txt`` / ``val.txt`` / ``test.txt`` lists.

    Args:
        root: the dataset root that holds the ``images`` / ``labels`` folders.
        out_dir: where the txt lists are written (defaults to ``root``).
        train_ratio: share of the images used for training.
        val_ratio: share of the images used for validation.
        seed: seed of the shuffle, so that the split is reproducible.
        images_dir_name: name of the folder holding the images.
        labels_dir_name: name of the folder holding the masks.
        mask_ext: extension of the mask files.

    Returns:
        dict with the ``train`` / ``val`` / ``test`` list paths and the counts.
    """
    root = os.path.abspath(os.path.expanduser(root))
    out_dir = os.path.abspath(os.path.expanduser(out_dir)) if out_dir else root

    images = collect_images(
        root, images_dir_name=images_dir_name, mask_ext=mask_ext, labels_dir_name=labels_dir_name
    )
    if not images:
        raise RuntimeError(f"no image/mask pair found under {root}")

    rng = random.Random(seed)
    rng.shuffle(images)

    total = len(images)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)
    if n_train == 0:
        n_train = 1
    if n_train + n_val > total:
        n_val = max(0, total - n_train)

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    lists = {}
    for name, paths in splits.items():
        if not paths:
            continue
        list_file = os.path.join(out_dir, f"{name}.txt")
        write_image_list(paths, list_file)
        lists[name] = list_file

    _logger_msg = (
        f"total={total} train={len(splits['train'])} val={len(splits['val'])} "
        f"test={len(splits['test'])} -> {out_dir}"
    )
    print(_logger_msg)
    return {
        "lists": lists,
        "counts": {k: len(v) for k, v in splits.items()},
        "total": total,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        "Build the yolo-style train/val/test txt lists of a segmentation dataset."
    )
    parser.add_argument("--root", type=str, required=True, help="dataset root")
    parser.add_argument("--out-dir", type=str, default="", help="where the lists are written")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-dir-name", type=str, default="images")
    parser.add_argument("--labels-dir-name", type=str, default="labels")
    parser.add_argument("--mask-ext", type=str, default=".png")
    args = parser.parse_args()

    create_dataset_lists(
        args.root,
        out_dir=args.out_dir or None,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
        images_dir_name=args.images_dir_name,
        labels_dir_name=args.labels_dir_name,
        mask_ext=args.mask_ext,
    )
