# sh tools/train.sh overflow yolov8n-seg.pt '' 20 64 640
task=$1
pretrain=$2
resume=$3
set_epochs=$4
batch_size=$5
img_size=$6
data_root=data/seg-dataset
date=$(date +%Y%m%d%H%M)
rm $data_root/$task/*.cache

yolo segment train data=$data_root/$task/dataset.yaml model=$pretrain epochs=$set_epochs batch=$batch_size \
    imgsz=$img_size device=0,1 project=ckpts name=$task/$date pretrained=True workers=8 patience=5 save_period=1
