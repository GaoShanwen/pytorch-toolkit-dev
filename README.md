# pytorch-toolkit-dev

building a multitask training platform based on Pytorch

## timm branch

### Motivation

1. Use SOTA AI-classfication toolkit.
2. Minimize our project.
3. Enhance codes' reusability.

### Quick Start

#### Install environment

the environments before install timm (these are not need be installed if you don't need to transfer to the rknn model.)

+ make=4.2
+ gcc=8.2
+ glib=2.29
+ python=3.8
+ rknn-tools2=1.5 py38

#### Install timm

```bash
<pytorch-cls-project> ~ $ # install timm and its environment(include rknn-tools2)
<pytorch-cls-project> ~ $ git clone git@github.com:GaoShanwen/pytorch-toolkit-dev.git
<pytorch-cls-project> ~ $ git checkout timm-dev
<pytorch-cls-project> ~ $ pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu102
```

### Training or validating owner data

```bash
<pytorch-cls-project> ~ $ # nohup train 4281 cls with 1k pretrain model; resize-256,crop-224,rand aa, re-0.2;
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=5,6 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=40401 tools/train.py \
        --dataset txt_data --data-dir dataset/exp-data/zero_dataset --model mobilenetv3_redution_large_100 -b 256 --epochs 60 --decay-epochs 2.4 \
        --sched cosine --decay-rate .973 --opt rmsproptf --opt-eps .001 -j 4 --warmup-lr 1e-5 --warmup-epochs 5 --weight-decay 1e-5 --model-ema \
        --model-ema-decay 0.9999 --aa rand-m9-mstd0.5 --scale 0.4 1. --remode pixel --reprob 0.2 --amp --lr-base .001875 --lr-noise 0.07 0.15 \
        --pretrained --num-classes 4281 &
<pytorch-cls-project> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --model mobilenetv3_redution_large_100.miil_in21k_ft_in1k
<pytorch-cls-project> ~ $ OMP_U_THREADS=1 MKL_NUM_THREADS=1 ... --pass-path dataset/exp-data/zero_dataset/pass_cats2.txt --num-classes 4091

<pytorch-cls-project> ~ $ # validate
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python tools/validate.py --dataset txt_data --data-dir dataset/exp-data/zero_dataset \
        --model mobilenetv3_redution_large_100 -b 256 -j 2 --img-size 224 --num-classes 4281 \
        --checkpoint output/train/20231019-183009-mobilenetv3_redution_large_100-224/model_best.pth.tar --crop-pct .875
<pytorch-cls-project> ~ $ # feat extracte
        OMP_U_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=7 python tools/feat_extract.py --dataset txt_data --data-dir dataset/exp-data/zero_dataset \
        --model mobilenetv3_redution_large_100 -b 256 -j 2 --img-size 224 --results-dir output/feats/mobilenet_large_100 --num-classes 4281 \
        --checkpoint output/train/20231019-183009-mobilenetv3_redution_large_100-224/model_best.pth.tar --crop-pct 1. --infer-mode train
<pytorch-cls-project> ~ $ # eval features
        python tools/eval_feats.py -g output/feats/mobilenet_large_100-train.npz -q output/feats/mobilenet_large_100-val.npz
```

#### Run model convert

```bash
<pytorch-cls-project> ~ $ # pth -> onnx
        python tools/scripts/onnx_export.py output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx \
        -m mobilenetv3_redution_large_100 --img-size 224 --opset 12 --num-classes 4281 \
        --checkpoint output/train/20231022-213614-mobilenetv3_redution_large_100-224/model_best.pth.tar 
<pytorch-cls-project> ~ $ # onnx -> rknn; validate(onnx and rknn, if model is cls model) model
        python tools/scripts/onnx2rknn.py output/converted_model/rk3566-mobilenetv3-224.rknn \
        --input output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx \
        --dataset txt_data --data-dir dataset/exp-data/zero_dataset --results-dir output/feats/mobilenet_large_100 \
        --model mobilenetv3_redution_large_100 -b 1 -j 1 --img-size 224 --results-dir output/feats/mobilenet_large_100 \
        --num-classes 4281 --mean 0.4850 0.4560 0.4060 --std 0.2290 0.2240 0.2250
<pytorch-cls-project> ~ $ # convert other
        ... output/converted_model/rk3566-regnety_016-224.rknn -m regnety_redution_016.tv2_in1k --results-dir output/feats/regnety_016 \
        --input output/converted_model/... --crop-pct 1.
```
