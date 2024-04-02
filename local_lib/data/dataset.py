######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: dataset.py
# function: create custom dataset(read imgs by txt-file).
######################################################
import io
import torch.utils.data as data
from torchvision import transforms

from timm.data import ImageDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import str_to_pil_interp

from .readers import create_reader


_CUSTOM_RAND_TFS = [
    'AutoContrast', 
    # 'Equalize',
    # 'Invert',
    'Rotate', 
    # 'Posterize', 
    'Solarize',
    # 'SolarizeAdd',
    'Color',
    'Contrast',
    'Brightness',
    'Sharpness',
    'ShearX', 
    'ShearY', 
    'TranslateXRel',
    'TranslateYRel',
    # 'Cutout'  # NOTE I've implement this as random erasing separately
]

class TxtReaderImageDataset(ImageDataset):
    def __init__(
        self,
        root,
        reader=None,
        split="train",
        class_map=None,
        load_bytes=False,
        img_mode="RGB",
        transform=None,
        target_transform=None,
        **kwargs
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(root=root, split=split, class_map=class_map, **kwargs)
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0

class CustomRandAADataset(data.Dataset):
    def __init__(self, dataset, input_size, auto_augment, interpolation, mean, set_cats=None):
        self.augmentation = None
        self.normalize = None
        self.dataset = dataset
        self.set_cats = set_cats
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.custom_aa = self.create_transform(input_size, auto_augment, interpolation, mean)

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.augmentation = x[1]
        self.normalize = x[2]

    @property
    def transform(self):
        return self.dataset.transform

    @transform.setter
    def transform(self, x):
        self._set_transforms(x)

    def _normalize(self, x):
        return x if self.normalize is None else self.normalize(x)

    def create_transform(self, input_size, auto_augment, interpolation='random', mean=IMAGENET_DEFAULT_MEAN):
        if isinstance(input_size, (tuple, list)):
            img_size_min = min(input_size[-2:])
        else:
            img_size_min = input_size
        aa_params = dict(
            translate_const=int(img_size_min * 0.45),
            img_mean=tuple([min(255, round(255 * x)) for x in mean]),
        )
        if interpolation and interpolation != 'random':
            aa_params['interpolation'] = str_to_pil_interp(interpolation)
        return transforms.Compose([rand_augment_transform(auto_augment, aa_params, _CUSTOM_RAND_TFS)])
    
    def __getitem__(self, i):
        x, y = self.dataset[i]  # all splits share the same dataset base transform
        # run the full augmentation on the remaining splits
        x = self.transform(x)
        if y in self.set_cats:
            x = self.custom_aa(x)
        else:
            x = self.augmentation(x)
        return self.normalize(x), y

    def __len__(self):
        return len(self.dataset)
    