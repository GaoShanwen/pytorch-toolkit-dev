#!/bin/bash
# sh tools/train.sh <data_name> <resume> <epochs> <batch_size> <img_size> <task> <model_size> <pretrained>
# example:
#   train from scratch:  sh tools/train.sh my_dataset '' 100 4 640 detect medium
#   resume training:     sh tools/train.sh my_dataset ckpts/my_dataset/last.ckpt 100 4 640 detect medium
#   custom pretrained:   sh tools/train.sh my_dataset '' 100 4 640 detect large /path/to/pretrained.ckpt
#
# model_size options: nano, small, medium (default), large, xlarge, 2xlarge
#   Note: xlarge and 2xlarge are segmentation-only models

data_name=$1
resume=$2
set_epochs=$3
batch_size=$4
img_size=$5
task=${6:-detect}
model_size=${7:-medium}

date=$(date +%Y%m%d%H%M)
data_root=data/$(echo "$task" | cut -c1-3)-dataset/$data_name
output_dir=ckpts/$task/$data_name/$date

export CUDA_DISABLE_NVML=1

device='0'
num_devices=$(echo $device | grep -o '[0-9]' | wc -l)

common_args="--data $data_root --epochs $set_epochs --batch $batch_size --imgsz $img_size --project ckpts --name $task/$data_name/$date --workers $num_devices --model $model_size"

if [ -z $resume ]; then
    rm -rf $output_dir
    pretrained=weights/rf-detr-$model_size.pth
    common_args="$common_args --pretrained $pretrained"
else
    echo "resume from $resume"
    common_args="$common_args --resume $resume"
fi
torchrun --nnodes=1 --nproc_per_node=$num_devices --master_port=40401 tools/train.py $common_args