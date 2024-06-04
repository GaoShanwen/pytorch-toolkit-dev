######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.12.08
# filenaem: dataset_factory.py
# function: create custom dataset.
######################################################
from timm.data import create_dataset

from local_lib.data.dataset import TxtReaderImageDataset, MultiLabelDataset
from local_lib.data.readers import ReaderImagePaths
from local_lib.data.readers.reader_for_reid import ReaderForReid


def create_custom_dataset(name, root, split="val", class_map=None, is_training=False, **kwargs):
    if name not in ["txt_data", "multilabel", "reid_data"]:
        return create_dataset(name, root, split, class_map, is_training, **kwargs)
    split = "train" if is_training else split.replace("validation", "val")
    assert split in ["train", "val", "infer"], f"split must be train/val or infer, but you set {split}"
    if name == "multilabel":
        multilabel = kwargs.get("multilabel", None)
        assert isinstance(multilabel, dict), "please set multilabel for args!"
        return MultiLabelDataset(root, split, multilabel.get("attributes", None), **kwargs)
    
    class_to_idx = kwargs.get("class_to_idx", None)
    reader = ReaderImagePaths(root, sort=False, class_to_idx=class_to_idx) if split == "infer" else None
    if name == "reid_data" and is_training:
        reader = ReaderForReid(root, split=split, class_map=class_map, **kwargs)
    return TxtReaderImageDataset(root, reader=reader, split=split, class_map=class_map, **kwargs)


if __name__ == "__main__":
    dataset_train = create_custom_dataset(
        "reid_data",
        root="dataset/optimize_task3",
        split="val",
        is_training=True,
        class_map=None,
        download=False,
        batch_size=64,
        seed=42,
        repeats=0.0,
        num_classes=2066,
        cats_path="dataset/optimize_task3/2066_cats.txt",
        multilabel=None,
        num_instance=4,
        world_size=8,
    )
    print(f"Loaded trainset: cats={len(dataset_train.reader.class_to_idx)}, imgs={len(dataset_train)}")
    from timm.data import create_loader
    # loader_train = create_loader(dataset_train, input_size=(3, 224, 224), batch_size=64)
    # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=64, sampler=sample)
    loader_train = create_loader(dataset_train, input_size=(3, 224, 224), batch_size=64)
    from local_lib.data.loader import rebuild_custom_loader
    loader_train = rebuild_custom_loader(loader_train, dataset_train.reader.sampler)
    for i, d in enumerate(loader_train):
        print(i, d[1])
        # if i <= 10:
        #     continue
        import pdb; pdb.set_trace()