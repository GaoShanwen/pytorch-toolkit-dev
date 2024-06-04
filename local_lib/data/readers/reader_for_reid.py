
from .reader_image_in_txt import ReaderImageTxt
from .reid_sampler import NaiveIdentitySampler


class ReaderForReid(ReaderImageTxt):
    def __init__(self, root, split, class_map="", **kwargs):
        super(ReaderForReid, self).__init__(root, split, class_map, **kwargs)
        targets = [target for _, target in self.samples]
        ids = sorted(self.class_to_idx.values())
        batch_size = kwargs.get('batch_size', None)
        num_instance = kwargs.get('num_instance', None)
        world_size = kwargs.get('world_size', None)
        self.sampler = NaiveIdentitySampler(targets, ids, batch_size, num_instance, world_size)
    