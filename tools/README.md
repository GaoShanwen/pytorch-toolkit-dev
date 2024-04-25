## Command Guides

<details open>

<summary>Train & Validate</summary>

- **Build Dataset**

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
    │   ├── environment.md
    │   └── structure.md
    ├── local_lib
    │   ├── data
    │   ├── models
    │   └── utils
    ├── output 
    │   └── train # .pth model path
    ├── README.md
    ├── requirements.txt
    └── tools
        ├── before
        ├── convert
        ├── eval_feats.py
        ├── post
        ├── train.py
        ├── validate.py
        └── visualize
```

- **Train Dataset**

```bash
<pytorch-toolkit-dev> ~$ # nohup train 4281 cls with 1k pretrain model; resize-256,crop-224,rand aa, re-0.2;
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -m torch.distributed.launch \
        --nproc_per_node=4 --master_port=40401 tools/train.py --config cfgs/base-regnety_redution_040.ra3_in1k.yaml \
        --options pretrained=True num_classes=663 data_dir=dataset/optimize_task3 cats_path=dataset/optimize_task3/664_cats.txt batch_size=64 log_wandb=True
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... --model mobilenetv3_redution_large_100.miil_in21k_ft_in1k
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... --config cfgs/removeredundancy/regnety_redution_040.ra3_in1k.yaml
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... --cats-path dataset/zero_dataset/save_cats2.txt
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... --options model-kwargs="reduction_dim=64"
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... --options amp=True amp_impl=apex 
```

- **Validate Dataset**

```bash
<pytorch-toolkit-dev> ~$ # run validate
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 python tools/validate.py \
        --config cfgs/base-regnety_redution_040.ra3_in1k.yaml --options num_gpu=2 \
        checkpoint=output/train/20240229-010436-regnety_redution_040_ra3_in1k-224/model_best.pth.tar \
        infer_mode=val num_classes=629 batch_size=128 data_dir=dataset/optimize_task3 cats_path=dataset/removeredundancy/629_cats.txt
<pytorch-toolkit-dev> ~$ # run recognize
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,6 python tools/infer_searching.py \
        --config cfgs/base-regnety_redution_040.ra3_in1k-infer.yaml --options num_gpu=2 \
        checkpoint=output/train/20231113-141942-regnety_redution_040_ra3_in1k-224/model_best.pth.tar \
        results_dir=output/feats/regnety_040 save_root=output/temp input_mode=file data_path=dataset/function_test/test.txt topk=30
<pytorch-toolkit-dev> ~$ # run search by gallery
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 python tools/infer_recognising.py \
        --config cfgs/base-regnety_redution_040.ra3_in1k-infer.yaml --options data_dir=dataset/blacklist2 batch_size=512 \
        checkpoint=output/train/blacklist2/model_best.pth.tar num_gpu=2 num_classes=663 infer_mode=val results_dir=output/vis/blacklist \
        input_mode=file data_path=dataset/blacklist2/val.txt cats_path=dataset/blacklist/10_cats.txt need_cats=dataset/blacklist/need_cats.txt
```

- **Feature Extracte & Eval**

```bash
<pytorch-toolkit-dev> ~$ # feat extracte
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=4,5,6,7 python tools/post/feat_extract.py \
        --config cfgs/base-regnety_redution_040.ra3_in1k.yaml --options model_classes=629 data_classes=10 \
        checkpoint=output/train/20231113-141942-regnety_redution_040_ra3_in1k-224/model_best.pth.tar \
        cats_path=dataset/blacklist2/10_cats.txt batch_size=512 num_gpu=4 results_dir=output/feats/blacklist \
        infer_mode=train data_dir=dataset/blacklist2
        model_classes=629 data_classes=175 \
        checkpoint=output/train/20240229-010436-regnety_redution_040_ra3_in1k-224/model_best.pth.tar \
        cats_path=dataset/function_test/test_1173/save_cats.txt batch_size=128 num_gpu=2 results_dir=output/feats/1173_o3 \
        infer_mode=train data_dir=dataset/function_test/test_1173
<pytorch-toolkit-dev> ~$ # eval features
        python tools/eval_feats.py -g output/feats/mobilenet_large_100-train.npz -q output/feats/mobilenet_large_100-val.npz
```

</details>

<details open>

<summary>Visualize Training Curve</summary>

- **Tensorboard**

Watch [tensorboard](http://localhost:6006/) curve in web after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # add tensorboard to train output:
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... tensorboard=logs
<pytorch-toolkit-dev> ~$ # view the output of tensorboard:
<pytorch-toolkit-dev> ~$ tensorboard --logdir logs/20231124-000720-regnety_redution_040_ra3_in1k-224
```

- **Wandb**

Watch [wandb](https://wandb.ai/) curve after run this commands.

```bash
<pytorch-toolkit-dev> ~$ # login 
<pytorch-toolkit-dev> ~$ wandb login
<pytorch-toolkit-dev> ~$ # add wandb to train output:
<pytorch-toolkit-dev> ~$ OMP_U_THREADS=1 ... log_wandb=true
```

</details>