# pytorch-toolkit-dev

building a multitask training platform based on Pytorch

## timm branch

### Motivation

1. Use State-of-the-Art AI-classfication toolkit.
2. Minimize our project.
3. Enhance codes' reusability.

### Quick Start

#### Install environment

the environments before install timm (these are not need be installed if you don't need to transfer to the rknn model.)
you can install these by this [document](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md).

+ make=4.2
+ gcc=8.2
+ glib=2.29
+ python=3.8
+ rknn-tools2=1.5 py38

#### Install timm

```bash
<pytorch-toolkit-dev> ~ $ # install timm and its environment(include rknn-tools2)
<pytorch-toolkit-dev> ~ $ git clone https://github.com/GaoShanwen/pytorch-toolkit-dev.git
<pytorch-toolkit-dev> ~ $ git checkout timm-dev
<pytorch-toolkit-dev> ~ $ pip install -r docs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu102
```

### Training or validating owner data

```bash
<pytorch-toolkit-dev> ~ $ # build dataset for train and validate
<pytorch-toolkit-dev> ~ $ ln -s /data/exp-data/* dataset/
<pytorch-toolkit-dev> ~ $ # nohup train 4281 cls with 1k pretrain model; resize-256,crop-224,rand aa, re-0.2;
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=40401 tools/train.py \
        --dataset txt_data --data-dir dataset/zero_dataset --model mobilenetv3_redution_large_100 -b 256 --epochs 60 --decay-epochs 2.4 \
        --sched cosine --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 4 --warmup-lr 1e-5 --warmup-epochs 5 --weight-decay 1e-5 --model-ema \
        --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --scale 0.4 1. --remode pixel --reprob 0.2 --amp --lr-base .001875 --lr-noise 0.07 0.15 \
        --pretrained --num-classes 4281 &
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --model mobilenetv3_redution_large_100.miil_in21k_ft_in1k
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --pass-path dataset/zero_dataset/pass_cats2.txt --num-classes 4091
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --cats-path dataset/zero_dataset/save_cats2.txt
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --model-kwargs reduction_dim=64

<pytorch-toolkit-dev> ~ $ # validate
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python tools/validate.py --dataset txt_data --data-dir dataset/zero_dataset \
        --model mobilenetv3_redution_large_100 -b 256 -j 2 --img-size 224 --num-classes 4281 \
        --checkpoint output/train/20231019-183009-mobilenetv3_redution_large_100-224/model_best.pth.tar --crop-pct .875
<pytorch-toolkit-dev> ~ $ # feat extracte
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python tools/post/feat_extract.py --dataset txt_data --data-dir dataset/zero_dataset \
        --model mobilenetv3_redution_large_100 -b 256 -j 2 --img-size 224 --results-dir output/feats/mobilenet_large_100 --num-classes 4281 \
        --checkpoint output/train/20231019-183009-mobilenetv3_redution_large_100-224/model_best.pth.tar --crop-pct 1. --infer-mode train

        CUDA_VISIBLE_DEVICES=5,6 python tools/post/feat_extract.py --dataset txt_data --data-dir dataset/removeredundancy --model regnety_redution_040.ra3_in1k \
        -b 256 -j 4 --img-size 224 --cats-path dataset/removeredundancy/save_cats.txt --pass-path '' --num-classes 629 --num-choose 0 629 \
        --checkpoint output/train/20231113-141942-regnety_redution_040_ra3_in1k-224/model_best.pth.tar --results-dir output/feats/regnety_040 \
        --no-prefetcher --num-gpu 2 --infer-mode train
<pytorch-toolkit-dev> ~ $ # eval features
        python tools/eval_feats.py -g output/feats/mobilenet_large_100-train.npz -q output/feats/mobilenet_large_100-val.npz
```

### Train with tensorboard or wandb

You can open this [tensorboard](http://localhost:6006/) or [wandb](https://wandb.ai/) in web to see after run this commands.

```bash
<pytorch-toolkit-dev> ~ $ # add tensorboard to train output:
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 ... --tensorboard logs
<pytorch-toolkit-dev> ~ $ # view the output of tensorboard:
<pytorch-toolkit-dev> ~ $ tensorboard --logdir logs/20231124-000720-regnety_redution_040_ra3_in1k-224

<pytorch-toolkit-dev> ~ $ # login 
<pytorch-toolkit-dev> ~ $ wandb login
<pytorch-toolkit-dev> ~ $ # add wandb to train output:
<pytorch-toolkit-dev> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 ... --log-wandb
```

### Convert pth model to onnx/rknn

```bash
<pytorch-toolkit-dev> ~ $ # pth -> onnx
        python tools/convert/onnx_export.py output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx \
        -m mobilenetv3_redution_large_100 --img-size 224 --opset 12 --num-classes 4281 \
        --checkpoint output/train/20231022-213614-mobilenetv3_redution_large_100-224/model_best.pth.tar 
<pytorch-toolkit-dev> ~ $ # onnx -> rknn; validate(onnx and rknn, if model is cls model) model
        python tools/convert/onnx2rknn.py output/converted_model/rk3566-mobilenetv3-224.rknn \
        --input output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx
<pytorch-toolkit-dev> ~ $ # convert other model
        ... output/converted_model/rk3566-regnety_016-224.rknn --input output/converted_model/... 
```

### normalize code

```bash
<pytorch-toolkit-dev> ~ $ # run this command after install black
<pytorch-toolkit-dev> ~ $ black --line-length=120 ./
```
