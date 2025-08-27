# sh tools/train.sh overflow yolov8n-seg.pt '' 100 256 640
# task=$1
data_name=$1
pretrain=$2
resume=$3
set_epochs=$4
batch_size=$5
img_size=$6
task=$7

data_root=data/$task-dataset
date=$(date +%Y%m%d%H%M)
rm $data_root/$data_name/*.cache
export OMP_NUM_THREADS=1


# yolo segment train data=$data_root/$data_name/dataset.yaml model=$pretrain epochs=$set_epochs batch=$batch_size \
#     imgsz=$img_size device=0,1 project=ckpts name=$data_name/$date workers=8 patience=0 #save_period=1 #find_unused_parameters=True pretrained=True --wandb

# yolo detect train data=$data_root/$data_name/dataset.yaml model=$pretrain epochs=$set_epochs batch=$batch_size \
#     imgsz=$img_size device=0,1 project=ckpts name=$data_name/$date workers=8 patience=0 #resume=true save_period=1 
if [ -z $pretrain ]; then
    torchrun --nproc_per_node=6 --master_port=40401 tools/train.py \
        --img $img_size --batch-size $batch_size --epochs $set_epochs --data $data_root/$task/dataset.yaml \
        --project ckpts --name $task/$date --patience 0 --device 2,3,4,5,6,7 --resume $resume # --entity jykj
else
    rm $data_root/$task/*.cache
    torchrun --nproc_per_node=6 --master_port=40401 tools/train.py \
        --img $img_size --batch-size $batch_size --epochs $set_epochs --data $data_root/$task/dataset.yaml \
        --project ckpts --name $task/$date --patience 0 --device 2,3,4,5,6,7 --weights $pretrain 
fi