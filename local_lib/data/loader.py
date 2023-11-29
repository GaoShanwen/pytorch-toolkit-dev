from timm.data import create_loader
import torch
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.transforms import Resize
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class OwnerResize(Resize):
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = cv2.resize(np.array(img), self.size, interpolation=cv2.INTER_CUBIC)
        return Image.fromarray(img)


def owner_transfrom():
    owner_trans = transforms.Compose([
        # transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC, antialias=False),
        OwnerResize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
    ])
    return owner_trans


def create_owner_loader(
        dataset,
        input_size,
        batch_size,
        is_training=False,
        use_prefetcher=True,
        no_aug=False,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_split=False,
        scale=None,
        ratio=None,
        hflip=0.5,
        vflip=0.,
        color_jitter=0.4,
        auto_augment=None,
        num_aug_repeats=0,
        num_aug_splits=0,
        interpolation='bilinear',
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_workers=1,
        distributed=False,
        crop_pct=None,
        crop_mode=None,
        collate_fn=None,
        pin_memory=False,
        fp16=False,  # deprecated, use img_dtype
        img_dtype=torch.float32,
        device=torch.device('cuda'),
        tf_preprocessing=False,
        use_multi_epochs_loader=False,
        persistent_workers=True,
        worker_seeding='all',
        mode_function='trainval',
):
    # if mode_function=='trainval':
    loader =  create_loader(
        dataset, input_size, batch_size, is_training, use_prefetcher,
        no_aug, re_prob, re_mode, re_count, re_split, scale, ratio,
        hflip, vflip, color_jitter, auto_augment, num_aug_repeats,
        num_aug_splits, interpolation, mean, std, num_workers,
        distributed, crop_pct, crop_mode, collate_fn, pin_memory,
        fp16, img_dtype, device, tf_preprocessing,
        use_multi_epochs_loader, persistent_workers, worker_seeding,
    )
    if mode_function=='feat_extract':
        loader.dataset.transform = owner_transfrom()
    return loader