# sh tools/train.sh clutterthings '' 100 192 640 segment
# sh tools/train.sh vehicle '' 100 12 640 detect
data_name=$1
resume=$2
set_epochs=$3
batch_size=$4
img_size=$5
task=$6

date=$(date +%Y%m%d%H%M)
# data_root=data/$task-dataset
data_root=data/$(echo "$task" | cut -c1-3)-dataset
export OMP_NUM_THREADS=1

if [ -z $resume ]; then
    rm $data_root/$data_name/*.cache
    if [ "$task" = "detect" ]; then
        pretrain=weights/yolov8m-coco.pt
    elif [ "$task" = "segment" ]; then
        pretrain=weights/yolov8n-seg.pt
    else
        echo task=$task error, only support 'detect' or 'segment'
    fi
    torchrun --nproc_per_node=4 --master_port=40401 tools/train.py \
        --data $data_root/$data_name/ultralytics.yaml --model $pretrain --task $task --project ckpts \
        --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
        --device 0,1,2,3 --options with_ignore=true amp=false #close_mosaic=20
else
    torchrun --nproc_per_node=4 --master_port=40401 tools/train.py \
        --imgsz $img_size --batch $batch_size --epochs $set_epochs --data $data_root/$data_name/ultralytics.yaml \
        --project ckpts --name $data_name/$date --patience 0 --device 0,1,2,3 --resume $resume --task $task # --entity jykj #save_period=1
fi