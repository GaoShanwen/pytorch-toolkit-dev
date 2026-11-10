######################################################
# author: gaowenjie
# email: gaoshanwen@bupt.cn
# date: 2026.11.09
# filenaem: seg_augmentation.py
# function: albumentations pipelines for segmentation train / val.
######################################################
import albumentations as A


def _check_size(img_size):
    """Round the size down to a multiple of 32 (encoder stride requirement)."""
    if isinstance(img_size, (tuple, list)):
        h, w = int(img_size[0]), int(img_size[1])
    else:
        h = w = int(img_size)
    h = max(32, (h // 32) * 32)
    w = max(32, (w // 32) * 32)
    return h, w


def get_train_augmentation(img_size=320, **kwargs):
    """Training pipeline: geometric jitter + photometric noise."""
    h, w = _check_size(img_size)
    return A.Compose(
        [
            A.Resize(h, w),
            A.HorizontalFlip(p=kwargs.get("hflip", 0.5)),
            A.VerticalFlip(p=kwargs.get("vflip", 0.0)),
            A.Affine(
                scale=(0.5, 1.5),
                translate_percent=0.1,
                rotate=(-15, 15),
                shear=(-5, 5),
                interpolation=cv2_interpolation(),
                mask_interpolation=cv2_nearest(),
                p=kwargs.get("affine_p", 0.5),
            ),
            A.ColorJitter(p=kwargs.get("color_jitter", 0.2)),
            A.GaussNoise(p=kwargs.get("gauss_noise", 0.2)),
        ]
    )


def get_val_augmentation(img_size=320, **kwargs):
    """Validation pipeline: resize only, so that the mask stays aligned."""
    h, w = _check_size(img_size)
    return A.Compose([A.Resize(h, w)])


def get_test_augmentation(img_size=320, **kwargs):
    """Test pipeline: resize only (same as validation)."""
    return get_val_augmentation(img_size, **kwargs)


def cv2_interpolation():
    import cv2

    return cv2.INTER_LINEAR


def cv2_nearest():
    import cv2

    return cv2.INTER_NEAREST


def build_augmentation(kind, img_size=320, **kwargs):
    """Factory dispatching to the requested augmentation pipeline.

    kind: one of ``train`` / ``val`` / ``test``.
    """
    kind = (kind or "train").lower()
    if kind == "train":
        return get_train_augmentation(img_size, **kwargs)
    if kind in ("val", "valid", "validation"):
        return get_val_augmentation(img_size, **kwargs)
    if kind in ("test", "infer"):
        return get_test_augmentation(img_size, **kwargs)
    raise ValueError(f"unknown augmentation kind: {kind}, expect train/val/test")
