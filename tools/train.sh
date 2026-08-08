# sh tools/train.sh clutterthings '' 100 192 640 segment
# sh tools/train.sh sensor '' 20 16 640 detect
# sh tools/train.sh trainval_set '' 100 16 640 pose
data_name=$1
resume=$2
set_epochs=$3
batch_size=$4
img_size=$5
task=$6

date=$(date +%Y%m%d%H%M)
data_root=data/$(echo "$task" | cut -c1-3)-dataset
# export OMP_NUM_THREADS=1
export CUDA_DISABLE_NVML=1 # disable nvml to avoid memory leak

device='0'
num_devices=$(echo $device | grep -o '[0-9]' | wc -l)
echo num_devices=$num_devices
if [ -z $resume ]; then
    rm $data_root/$data_name/*.cache
    if [ "$task" = "detect" ]; then
        pretrain=weights/yolo26s.pt
        torchrun --nnodes=$num_devices --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/dataset.yaml --model $pretrain --task $task --project ckpts \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device #--options amp=false # close_mosaic=20 cos_lr=true 
        # yolo train detect data=$data_root/$data_name/dataset.yaml batch=$batch_size epochs=$set_epochs \
        #     device=$device task=$task project=ckpts name=$data_name/$date model=$pretrain
    elif [ "$task" = "pose" ]; then
        data_root=data/$(echo "$task" | cut -c1-4)-dataset
        rm $data_root/$data_name/*.cache $data_root/$data_name/*/*.cache
        pretrain=weights/yolo26s-pose.pt
        torchrun --nnodes=$num_devices --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/dataset.yaml --model $pretrain --task $task --project lingxin \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device --workers $num_devices --options \
            mixed_data=True mixed_alpha=0.06 class_mapping={8:9,10:11} \
            symmetry_match=True symmetry_categories=[4,5,6,7] symmetry_pairs=[[1,2],[3,5],[4,6]] \
            quad_eiou=True quad_categories=[5,6,7,8,9,10,11,12,13,14] quad_indices=[3,4,5,6] 
            #func=tune space="{lr0:(5e-4,0.01)}" #,mixed_alpha:(0.06,0.15,'uniform')
    else
        echo task=$task error, only support 'detect' or 'pose'
    fi
else
    date=$(echo $resume | cut -d'/' -f5 | cut -d'.' -f4)
    echo "resume from $resume, date=$date"
    if [ "$task" = "detect" ]; then
        torchrun --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/dataset.yaml --model $resume --task $task --project ckpts \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device --resume --options amp=false # --entity jykj # --entity jykj #save_period=1
    elif [ "$task" = "pose" ]; then
        torchrun --nnodes=$num_devices --nproc_per_node=$num_devices --master_port=40401 tools/train.py \
            --data $data_root/$data_name/dataset.yaml --model $resume --task $task --project ckpts \
            --name $data_name/$date --epochs $set_epochs --patience 0 --imgsz $img_size --batch $batch_size \
            --device $device --resume --workers $num_devices --options mixed_data=True mixed_alpha=0.06 \
            symmetry_match=True symmetry_categories=[4,5,6,7] symmetry_pairs=[[1,2],[3,5],[4,6]]
    else
        echo task=$task error, only support 'detect' or 'pose'
    fi
fi