## Command Guides

**Train & Validate**

<details close>

<summary> Build Dataset </summary>

Run this comand, make sure your folder format is follow:

```bash
<pytorch-toolkit-dev> ~$ # build dataset for train and validate
<pytorch-toolkit-dev> ~$ ln -s /data/exp-data/* dataset/
```

```bash
<pytorch-toolkit-dev> <timm-dev branch>
    ├── cfgs
    │   └── base-regnety_redution_040.ra3_in1k.yaml
    ├── dataset
    │   ├── 1.jpg
    │   ├── <your datasets> # train/validate data
    │   ├── simsun.ttc # chinese fonts
    │   └── README.md
    ├── docs
    │   ├── demo4reid.jpg
    │   ├── INSTALL.md
    │   ├── model-zoo
    │   └── requirements.txt
    ├── local_lib
    │   ├── data
    │   ├── models
    │   └── utils
    ├── output 
    │   └── ckpts # .pth model path
    ├── README.md
    └── tools
        ├── before
        ├── deploy
        ├── post
        ├── README.md
        ├── eval_feats.py
        ├── train.py
        └── validate.py
```

</details>

<details open>

<summary> Train Dataset </summary>

```bash
<pytorch-toolkit-dev> ~$ # nohup train 4281 cls with 1k pretrain model; resize-256,crop-224,rand aa, re-0.2;
        torchrun --nproc_per_node=8 --master_port=40401 tools/train.py --config cfgs/base-regnety_redution_040.ra3_in1k.yaml
<pytorch-toolkit-dev> ~$ torchrun ... --config cfgs/removeredundancy/regnety_redution_040.ra3_in1k.yaml
<pytorch-toolkit-dev> ~$ torchrun ... --options model=mobilenetv3_redution_large_100.miil_in21k_ft_in1k
<pytorch-toolkit-dev> ~$ torchrun ... --options model-kwargs="reduction_dim=64"
<pytorch-toolkit-dev> ~$ torchrun ... --options log_wandb=true
```

</details>

<details open>

<summary> Validate Dataset </summary>

```bash
<pytorch-toolkit-dev> ~$ # run validate
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/validate.py --config <cfgs/your/config/for/val/xx.yaml> \
        --options checkpoint=<your/model/path/xx.pth.tar> 
<pytorch-toolkit-dev> ~$ # run search by gallery
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/infer_searching.py --config <cfgs/your/config/for/infer/xx-infer.yaml> \
        --options checkpoint=<your/model/path/xx.pth.tar> topk=30
<pytorch-toolkit-dev> ~$ # run recognize
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/infer_recognising.py --config <cfgs/your/config/for/infer/xx-infer.yaml> \
        --options checkpoint=<your/model/path/xx.pth.tar>
```

- **Feature Extracte & Eval**

```bash
<pytorch-toolkit-dev> ~$ # feat extracte
        CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python tools/post/feat_extract.py --config <cfgs/your/config/for/infer/xx-infer.yaml> \
        --options model_classes=629 data_classes=10 checkpoint=<your/model/path/xx.pth.tar> \
        batch_size=512 results_dir=output/feats/blacklist infer_mode=train 
<pytorch-toolkit-dev> ~$ # eval features
        python tools/eval_feats.py -g output/feats/mobilenet_large_100-train.npz -q output/feats/mobilenet_large_100-val.npz
<pytorch-toolkit-dev> ~$ # eval features in sql
        python tools/eval_sql.py --set-date 2024-07-27 --brand-id 1386 
```

</details>

<details close>

<summary>Visualize Training Curve</summary>

- **Tensorboard**

Watch [tensorboard](http://localhost:6006/) curve in web after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # add tensorboard dir to train output:
<pytorch-toolkit-dev> ~$ CUDA_VISIBLE_DEVICES=0, ... --options tensorboard=logs
<pytorch-toolkit-dev> ~$ # view the output of tensorboard:
<pytorch-toolkit-dev> ~$ tensorboard --logdir logs/202311xx-000720-regnety_redution_040_ra3_in1k-224
```

- **Wandb**

Watch [wandb](https://wandb.ai/) curve after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # login 
<pytorch-toolkit-dev> ~$ wandb login
<pytorch-toolkit-dev> ~$ # add wandb to train output:
<pytorch-toolkit-dev> ~$ CUDA_VISIBLE_DEVICES=0, ... --options log_wandb=true
```

</details>

<details open>

TODO:
 - **1** merge eval_feats and eval_sql
 - **2** try reid example again
 - **3** add eval/draw function for reid
 - **4** add resume function for wandb
 - **5** add logger function for local-lib

</details>