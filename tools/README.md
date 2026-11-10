## Command Guides

**Train & Evaluate**

<details close>

<summary> Prepare Dataset </summary>

The segmentation dataset follows the **yolo** convention: a txt list holds one
image path per line, and the mask path is derived by replacing the `images`
folder of the image path by `labels` (and the extension by `mask_ext`).

```bash
<pytorch-toolkit-dev> ~$ # link your dataset into the repo
<pytorch-toolkit-dev> ~$ ln -s /data/exp-data/seg-dataset dataset/
```

```bash
<pytorch-toolkit-dev>
    ├── dataset
    │   ├── seg-dataset            # your data (symlink)
    │   │   ├── images             # 0001.jpg, 0002.jpg ...
    │   │   ├── labels             # 0001.png, 0002.png ...
    │   │   ├── train.txt          # one image path per line
    │   │   ├── val.txt
    │   │   └── test.txt
    │   ├── seg-template.yaml      # config template
    │   └── README.md
    ├── local_lib
    │   ├── data                   # dataset / augmentation / dataloader
    │   ├── models                 # smp model / loss / optimizer factories
    │   └── utils                  # yaml config loader / logger / metrics
    ├── output
    │   └── ckpts                  # .pth checkpoints
    ├── README.md
    └── tools
        ├── README.md
        ├── create_lists.py
        ├── train.py
        ├── evaluate.py
        └── validate.py
```

Build the `train.txt` / `val.txt` / `test.txt` lists from the folder tree:

```bash
<pytorch-toolkit-dev> ~$ python tools/create_lists.py --root dataset/seg-dataset \
        --train-ratio 0.8 --val-ratio 0.1
<pytorch-toolkit-dev> ~$ # train: 16 images  val: 2 images  test: 2 images -> dataset/seg-dataset
```

The lists can also be written by hand; the paths may be absolute or relative to
the folder of the list file:

```bash
<pytorch-toolkit-dev> ~$ head -2 dataset/seg-dataset/train.txt
/data/exp-data/seg-dataset/images/a/0001.jpg
/data/exp-data/seg-dataset/images/b/0002.jpg
```

</details>

<details close>

<summary> Train Dataset </summary>

```bash
<pytorch-toolkit-dev> ~$ # train with the template config
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml
<pytorch-toolkit-dev> ~$ # override any config entry on the command line
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options epochs=50 batch_size=16 data.root=dataset/seg-dataset

<pytorch-toolkit-dev> ~$ # switch the architecture / encoder
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options model=FPN encoder_name=resnet50

<pytorch-toolkit-dev> ~$ # set the wandb project
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options wandb.project=my-seg-project

<pytorch-toolkit-dev> ~$ # log the curves to tensorboard
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options tensorboard=logs
<pytorch-toolkit-dev> ~$ tensorboard --logdir logs
```

- **Config template**

See [dataset/seg-template.yaml](../dataset/seg-template.yaml) for a fully
documented example. The most important entries are:

| key | meaning |
| --- | --- |
| `classes` | list of class names, index 0 is the background |
| `data.root` | dataset root holding the `train.txt` / `val.txt` lists |
| `data.name` | name of the dataset, added to the output folder (empty takes it from `data.root`) |
| `data.train_list` / `data.val_list` | explicit txt lists (optional) |
| `data.train_split` / `data.val_split` | split names of the lists (`train` / `val`) |
| `data.mask_ext` | extension of the mask files (`.png` by default) |
| `data.images_dir_name` / `data.labels_dir_name` | folder names swapped to derive the mask path |
| `model` / `encoder_name` | smp architecture and backbone |
| `loss` | `dice` / `jaccard` / `focal` / `ce` / `bce` |
| `optimizer` / `lr` / `scheduler` / `epochs` | optimization settings |
| `img_size` | training resolution, must be a multiple of 32 |
| `output` | the checkpoints are written to `<output>/<dataset name>/<experiment>` |
| `log_wandb` / `wandb.*` / `tensorboard` | experiment logging |

</details>

<details close>

<summary> Evaluate Dataset </summary>

```bash
<pytorch-toolkit-dev> ~$ # evaluate the best checkpoint on the val split
<pytorch-toolkit-dev> ~$ python tools/evaluate.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth

<pytorch-toolkit-dev> ~$ # write the metrics to a json file
<pytorch-toolkit-dev> ~$ python tools/evaluate.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options results_file=output/results.json
```

The evaluation prints the loss, the IoU, the mIoU and the IoU of every class.

</details>

<details close>

<summary> Run the Inference </summary>

`tools/inference.py` runs a checkpoint on images and writes a picture with the
two halves next to each other:

- **a split of the dataset** (the val split by default): the left half holds
  the **label** and the right half the **prediction**;
- **a single image without a label**: the left half holds the **original
  image** and the right half the **prediction**.

The source of the images is picked in this order: `inference.image` (a single
image), `inference.source` (a folder or a glob), then the split of the
dataset (`inference.split`).

```bash
<pytorch-toolkit-dev> ~$ # the val split of the dataset (every image)
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth

<pytorch-toolkit-dev> ~$ # the test split of the dataset
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options inference.split=test

<pytorch-toolkit-dev> ~$ # a single image without a label
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options inference.image=/path/to/photo.jpg

<pytorch-toolkit-dev> ~$ # a folder of images without a label
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options inference.source=/path/to/images

<pytorch-toolkit-dev> ~$ # a glob pattern
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options "inference.source=/path/to/images/*.jpg"

<pytorch-toolkit-dev> ~$ # where the pictures are written
<pytorch-toolkit-dev> ~$ python tools/inference.py --config dataset/seg-template.yaml \
        --checkpoint output/ckpts/<dataset name>/<exp>/best.pth \
        --options inference.source=/path/to/images inference.out_dir=output/my_vis
```

The split of the dataset is looked up in this order:
`data.<split>_list`, then `data.train_list` / `data.val_list`, then
`<data.root>/<split>.txt`. A split that is not listed (e.g. `test`) falls
back to the train list, so add a `data.test_list` to the config to use it.

The mask of a class is **blended** into the picture: the pixels of the mask
are 70% of the colour of the class and 30% of the original pixels.

The images are read and resized **exactly like in the validation of the
training** (the `img_size` of the config) and inferred **by batches** (the
`batch_size` of the config), so the inference sees the same kind of input as
the training. The prediction is then resized back to the size of the original
picture before it is drawn.

The drawing itself lives in `local_lib/utils/visualize.py`, so a picture can
also be built from a python script:

```python
from local_lib.utils.visualize import make_figure, overlay_prediction, save_image

figure = make_figure(image, pred, label, classes, settings)
save_image("out.jpg", figure)
```

Every setting is read from the **`inference`** section of the yaml config, so
the config of the training is used as it is:

| key | default | meaning |
| --- | --- | --- |
| `inference.checkpoint` | `''` | the checkpoint to load; empty uses the one of `--checkpoint` |
| `inference.out_dir` | `output/vis` | the folder the pictures are written to |
| `inference.image` | `''` | a single image to infer (without a label) |
| `inference.source` | `''` | a folder or a glob of images to infer (without a label) |
| `inference.split` | the val split | the split that is inferred when no image is given |
| `inference.ratio` | `0.7` | the share of the colour of the mask (the rest is the original) |
| `inference.classes` | `''` | the ids of the classes that are drawn; empty draws every class but the background |
| `inference.palette` | `[]` | the BGR colour of every class; empty uses the default one |
| `inference.img_size` | the `img_size` of the training | the size the images are resized to before the inference |
| `inference.batch_size` | the `batch_size` of the training | the number of images that are inferred at once |
| `inference.workers` | the `workers` of the training | the number of workers that read the images |
| `inference.labels` | `[]` | the titles of the two halves; empty uses `label`/`image` and `prediction` |
| `inference.gap` | `8` | the width of the band between the two halves, in pixels |
| `inference.suffix` | `''` | a suffix added to the name of the written pictures |
| `inference.quality` | `95` | the quality of the written jpg (1-100) |

</details>

<details close>

<summary>Visualize Training Curve</summary>

- **Tensorboard**

Watch [tensorboard](http://localhost:6006/) curve in web after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # add tensorboard dir to train output:
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options tensorboard=logs
<pytorch-toolkit-dev> ~$ # view the output of tensorboard:
<pytorch-toolkit-dev> ~$ tensorboard --logdir logs/<exp>
```

- **Wandb**

Watch [wandb](https://wandb.ai/) curve after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # login
<pytorch-toolkit-dev> ~$ wandb login
<pytorch-toolkit-dev> ~$ # train (the wandb logger is on by default)
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml
<pytorch-toolkit-dev> ~$ # turn it off for a run
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml \
        --options log_wandb=false
```

The logger is **on by default**. It is disabled with `log_wandb: false` in
the config or with `--options log_wandb=false` on the command line; the
`--wandb` flag forces it on.
The `wandb` section of the config holds the settings:

| key | default | meaning |
| --- | --- | --- |
| `project` | the git branch | the name of the project |
| `entity` / `group` | `''` | the user / the team and the group; empty uses the ones of the api key |
| `job_type` | `seg-train` | the job type |
| `mode` | `online` | `online` / `offline` / `disabled` |
| `api_key` | `''` | the key; when empty the `WANDB_API_KEY` env var or `wandb login` is used |
| `base_url` | `https://api.bandw.top` | the wandb server (see below) |
| `http_timeout` | `60` | the timeout of the http requests, in seconds |
| `observability` | `false` | send the system metrics (gpu / cpu) |
| `offline_fallback` | `true` | fall back to the offline mode when the api cannot be reached |
| `watch` | `gradients` | what `wandb.watch` tracks: `gradients` / `parameters` / `all` |
| `watch_freq` | `1000` | how often the gradients are logged |

**The name of the project.** It is the name of the current **git branch**, so
the runs of the different branches do not get mixed up. `project: branch`
(the default) uses the branch name alone, a fixed name prefixes it
(`<name>-<branch>`) and an empty value uses the branch alone.

```bash
<pytorch-toolkit-dev> ~$ git switch feature/unet
<pytorch-toolkit-dev> ~$ python tools/train.py --config dataset/seg-template.yaml
# -> the project is 'feature-unet'
```

When the api key is missing the run is **not** started: a warning is printed
and the training goes on without any wandb log.

**The wandb server.** The official `https://api.wandb.ai` refuses the
connections coming from china with a `403 Forbidden` (the html page of the
google edge is returned, so it is not an authentication problem). The default
is therefore the mirror `https://api.bandw.top`, which forwards the requests
to the official server. Set `base_url` to `https://api.wandb.ai` to use the
official one directly, or export `WANDB_BASE_URL`.

When the configured server answers `403`, the run is **automatically retried
on the mirror**, so a config that still points at the official server keeps
working.

The mirror is a free relay and it is often slower than the official server.
The wandb service answers an api request with a hardcoded timeout of 10
seconds, so the request fails with *"the service process is busy and did not
respond in time"*. The timeout is raised to 60 seconds with
`http_timeout` (the `WANDB_HTTP_TIMEOUT` env var does the same).

The api key is the one of the **official** server, the mirror does not have
its own accounts:

```bash
<pytorch-toolkit-dev> ~$ wandb login --host https://api.bandw.top
```

The key is stored in `~/.netrc` and the server in
`~/.config/wandb/settings`, so it is picked up by every new shell.

If the wandb api cannot be reached at all, the run falls back to the
**offline mode** and the logs stay on the disk; they are uploaded later with:

```bash
<pytorch-toolkit-dev> ~$ wandb sync wandb/offline-run-*
```

The system metrics are exported by an opentelemetry exporter that runs in the
wandb service; it fails on its own when the api is unreachable, so it is
disabled by default (`observability: false`).

</details>

<details open>

TODO:
 - **1.** add distributed training support (torchrun / DDP)
 - **2.** add gradient accumulation and larger batch training
 - **3.** add TTA (test-time augmentation) to the evaluation
 - **4.** add ONNX export for the segmentation models
 - **5.** add Hugging Face Hub support

</details>
