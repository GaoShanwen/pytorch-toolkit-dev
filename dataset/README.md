# Dataset

This folder holds the datasets and the configuration templates.

## Layout

The segmentation data follows the **yolo** convention: a txt list holds one
image path per line, and the mask path is derived by replacing the `images`
folder of the image path by `labels` (and the extension by `mask_ext`).

```bash
dataset
├── seg-dataset            # your data (usually a symlink)
│   ├── images             # 0001.jpg, 0002.jpg ...
│   ├── labels             # 0001.png, 0002.png ...
│   ├── train.txt          # one image path per line
│   ├── val.txt
│   └── test.txt
├── seg-template.yaml      # the documented config template
└── README.md
```

So `/data/seg/images/a/0001.jpg` is paired with `/data/seg/labels/a/0001.png`.

Link your dataset into the repo and build the lists:

```bash
ln -s /data/exp-data/seg-dataset dataset/
python tools/create_lists.py --root dataset/seg-dataset --train-ratio 0.8 --val-ratio 0.1
```

The lists can also be written by hand; the paths may be absolute or relative to
the folder of the list file, and empty lines / `#` comments are skipped.

## Config template

[seg-template.yaml](seg-template.yaml) documents every option of the training
and evaluation scripts. Copy it and adapt it to your own dataset:

```bash
cp dataset/seg-template.yaml dataset/my-seg.yaml
```

Run it with:

```bash
python tools/train.py --config dataset/my-seg.yaml
```
