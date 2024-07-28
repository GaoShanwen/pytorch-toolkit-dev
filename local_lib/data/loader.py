######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: loader.py
# function: custom dataloader, to create the same input with rknn.
######################################################
import cv2
import numpy as np
import torch
from PIL import Image
from timm.data import create_loader
from torchvision import transforms
from torchvision.transforms import Resize
from torch.utils.data import DataLoader
from timm.data.loader import MultiEpochsDataLoader, PrefetchLoader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

class CustomResize(Resize):
    def __init__(self, size, interpolation):
        self.size = size
        self.interpolation = interpolation

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be scaled.

        Returns:
            PIL Image or Tensor: Rescaled image.
        """
        img = cv2.resize(np.array(img), self.size, interpolation=self.interpolation)
        return Image.fromarray(img)

    def __repr__(self):
        interpolate_str = "cv2.INTER_CUBIC"
        return self.__class__.__name__ + f"(size={self.size}, interpolation={interpolate_str})"


def custom_transfrom(input_size=[3, 224, 224]):
    img_size = input_size[1:]
    custom_trans = transforms.Compose(
        [
            CustomResize(img_size, interpolation=cv2.INTER_CUBIC),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225]),
            ),
        ]
    )
    return custom_trans


def create_custom_loader(dataset, input_size, batch_size, transfrom_mode="trainval", **kwargs):
    loader = create_loader(dataset, input_size, batch_size, **kwargs)
    if transfrom_mode == "custom":
        if input_size is None:
            loader.transform = custom_transfrom()
        else:
            loader.transform = custom_transfrom(input_size)
    print(loader.transform)
    return loader


def rebuild_custom_loader(
        loader, 
        sampler, 
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        channels=3,
        re_prob=0.,
        re_mode='const',
        re_count=1,
        re_num_splits=0,
    ):
    config_loader = loader.loader
    loader_class = DataLoader if isinstance(config_loader, DataLoader) else MultiEpochsDataLoader

    loader_args = dict(
        batch_size=config_loader.batch_size,
        shuffle=False,
        num_workers=config_loader.num_workers,
        sampler=sampler,
        collate_fn=config_loader.collate_fn,
        pin_memory=config_loader.pin_memory,
        drop_last=True, #config_loader.drop_last,
        worker_init_fn=config_loader.worker_init_fn,
        persistent_workers=config_loader.persistent_workers
    )
    dataset = config_loader.dataset
    try:
        custom_loader = loader_class(dataset, **loader_args)
    except TypeError as e:
        loader_args.pop('persistent_workers')  # only in Pytorch 1.7+
        custom_loader = loader_class(dataset, **loader_args)
    if isinstance(loader, PrefetchLoader):
        custom_loader = PrefetchLoader(
            custom_loader,
            mean=mean,
            std=std,
            channels=channels,
            device=loader.device,
            fp16=loader.img_dtype==torch.float16,  # deprecated, use img_dtype
            img_dtype=loader.img_dtype,
            re_prob=re_prob,
            re_mode=re_mode,
            re_count=re_count,
            re_num_splits=re_num_splits
        )
    return custom_loader

if __name__ == "__main__":
    img_path = "dataset/test_imgs/1.jpg"
    input = open(img_path, "rb")  # data_trans().unsqueeze(0)
    trans_resize = CustomResize([224, 224])
    res_img = trans_resize(input)
