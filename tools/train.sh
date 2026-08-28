#!/bin/bash
# sh tools/train.sh <data_name> <resume> <epochs> <batch_size> <img_size> <task> <model_size> <pretrained>
# example:
#   train from scratch:  sh tools/train.sh my_dataset '' 100 4 640 detect medium
#   resume training:     sh tools/train.sh my_dataset ckpts/my_dataset/last.ckpt 100 4 640 detect medium
#   custom pretrained:   sh tools/train.sh my_dataset '' 100 4 640 detect large /path/to/pretrained.ckpt
#
# model_size options: nano, small, medium (default), large, xlarge, 2xlarge
#   Note: xlarge and 2xlarge are segmentation-only models

data_name=${1:-BakingRecognizeCOCO}
resume=${2:-''}
set_epochs=${3:-100}
batch_size=${4:-4}
img_size=${5:-576}
task=${6:-detect}
model_size=${7:-medium}

date=$(date +%Y%m%d%H%M)
data_root=data/$(echo "$task" | cut -c1-3)-dataset/$data_name
output_dir=ckpts/$task/$data_name/$date

export CUDA_DISABLE_NVML=1

device='0'
export CUDA_VISIBLE_DEVICES="$device"
compact="$(echo "$device" | tr -d ' ')"
if [ -z "$compact" ]; then
    num_devices=1
else
    num_devices=$(echo "$compact" | awk -F',' '{print NF}')
fi

device_arg="cpu"
if [ -n "$device" ] && [ "$device" != "-1" ]; then
    case "$compact" in *,*) device_arg="cuda" ;;
        *) device_arg="cuda:${compact}" ;;
    esac
fi

common_args="--data $data_root --epochs $set_epochs --batch $batch_size --imgsz $img_size \
--project ckpts --name $task/$data_name/$date --workers 4 --model $model_size --device $device_arg"

options_args="--options rfdetr_model_size=medium class_mapping={8:9,10:11} \
mixed_alpha=0.06 background_data=data/pose-dataset/BakingRecognize/plubic.txt"
if [ -z $resume ]; then
    rm -rf $output_dir
    pretrained=$(pwd)/weights/rf-detr-$model_size.pth
    common_args="$common_args --pretrained $pretrained"
else
    echo "resume from $resume"
    common_args="$common_args --resume $resume"
fi

torchrun --nnodes=1 --nproc_per_node=$num_devices --master_port=40401 tools/train.py $common_args $options_args
