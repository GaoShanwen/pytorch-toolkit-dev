# sh tools/train.sh clutterthings '' 100 192 640 segment
# sh tools/train.sh vehicle '' 100 112 640 detect
data_name=$1
resume=$2
set_epochs=$3
batch_size=$4
img_size=$5
task=$6

date=$(date +%Y%m%d%H%M)
data_root=data/$(echo "$task" | cut -c1-3)-dataset
# export OMP_NUM_THREADS=1

device='0,1'
num_devices=$(echo $device | grep -o '[0-9]' | wc -l)
if [ -z $resume ]; then
    rm $data_root/$data_name/*.cache
    if [ "$task" = "detect" ]; then
        pretrain=weights/yolov8m-coco.pt
        # pretrain=weights/yolo11m.pt
        torchrun --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/ultralytics.yaml --model $pretrain --task $task --project ckpts \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device --options with_ignore=true amp=false # close_mosaic=20 cos_lr=true 
    elif [ "$task" = "segment" ]; then
        pretrain=weights/yolov8n-seg.pt
        torchrun --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/ultralytics.yaml --model $pretrain --task $task --project ckpts \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device --options amp=false # close_mosaic=20 cos_lr=true 
    else
        echo task=$task error, only support 'detect' or 'segment'
    fi
else
    torchrun --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
        --data $data_root/$data_name/ultralytics.yaml --model $resume --task $task --project ckpts \
        --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
        --device $device --resume --options with_ignore=true amp=false # --entity jykj # --entity jykj #save_period=1
fi