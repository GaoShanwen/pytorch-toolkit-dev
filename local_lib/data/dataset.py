######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: dataset.py
# function: create custom dataset(read imgs by txt-file).
######################################################
import os
import csv
from PIL import Image
import torch.utils.data as data
from torchvision import transforms

from timm.data import ImageDataset
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.auto_augment import rand_augment_transform
from timm.data.transforms import str_to_pil_interp
from timm.utils.misc import natural_key

from .readers import create_reader
from .custom_aa import _CUSTOM_RAND_TFS

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


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, split="train", attributes=[], transform=None, **kwargs):
        super().__init__()
        self.transform = transform
        self.attributes = attributes
        self.data, self.annos = [], []
        self.label_to_idx = {}
        for attr in attributes:
            attr_annos = os.path.join(root, f"{attr}.txt")
            assert os.path.exists(attr_annos), "make sure the attribute annotation file exists!"
            with open(attr_annos, "r") as f:
                unique_labels = list(sorted(set(cat.strip() for cat in f.readlines()), key=natural_key))
                self.label_to_idx.update({attr: {cat:i for i, cat in enumerate(unique_labels)}})

        # read the annotations from the CSV file
        with open(os.path.join(root, f"{split}.csv")) as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.data.append(row['image_path'])
                self.annos.append({attr: self.label_to_idx[attr][row[attr]] for attr in attributes})

    def __getitem__(self, idx):
        # take the data sample by its index
        img_path = self.data[idx]
        # read image
        img = Image.open(img_path)

        # apply the image augmentations if needed
        if self.transform:
            img = self.transform(img)
        labels = self.annos[idx]
        return img, labels

    def __len__(self):
        return len(self.data)

class CustomRandAADataset(data.Dataset):
    def __init__(self, dataset, input_size, auto_augment, interpolation, mean, convert_epoch):
        self.augmentation = None
        self.default_aa = None
        self.normalize = None
        self.dataset = dataset
        self.convert_epoch = convert_epoch
        if self.dataset.transform is not None:
            self._set_transforms(self.dataset.transform)
        self.custom_aa = self.create_transform(input_size, auto_augment, interpolation, mean)

    def set_epoch(self, count):
        # TFDS and WDS need external epoch count for deterministic cross process shuffle
        if hasattr(self, 'reader') and hasattr(self.reader, 'set_epoch'):
            self.reader.set_epoch(count)
        self.augmentation = self.default_aa if count < self.convert_epoch else self.custom_aa

    def _set_transforms(self, x):
        assert isinstance(x, (list, tuple)) and len(x) == 3, 'Expecting a tuple/list of 3 transforms'
        self.dataset.transform = x[0]
        self.default_aa = x[1]
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
        x = self.augmentation(self.transform(x))
        return self.normalize(x), y

    def __len__(self):
        return len(self.dataset)
    