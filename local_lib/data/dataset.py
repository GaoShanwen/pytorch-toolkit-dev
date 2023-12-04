from .readers import create_reader
from timm.data import ImageDataset


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
        **kwargs,
    ):
        if reader is None or isinstance(reader, str):
            reader = create_reader(
                root=root,
                split=split,
                class_map=class_map,
                **kwargs,
            )
        self.reader = reader
        self.load_bytes = load_bytes
        self.img_mode = img_mode
        self.transform = transform
        self.target_transform = target_transform
        self._consecutive_errors = 0
