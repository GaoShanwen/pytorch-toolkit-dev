# Usage:
#   sh tools/train.sh <data_name> <resume> <epochs> <batch_size> <img_size> <task>
# Examples:
#   sh tools/train.sh Person '' 420 256 192 pretrain
#   sh tools/train.sh Person '' 420 256 192 pretrain_4kp
#   sh tools/train.sh Person ckpts/xxx/epoch_100.pth 420 256 192 pretrain
# Tasks:
#   pretrain      - pretrain RTMPose-m with 17 keypoints (default)
#   pretrain_4kp  - pretrain RTMPose-m with 4 keypoints + 10 classes (filtered)
#   pretrain_s    - pretrain RTMPose-s with 17 keypoints
#   finetune      - finetune on specific dataset

data_name=$1
resume=$2
set_epochs=$3
batch_size=$4
img_size=$5
task=$6

# Compute input_size from img_size (width=img_size, height=img_size*4/3 for 3:4 portrait)
input_width=$img_size
input_height=$img_size #$((input_width * 4 / 3))
featmap_width=$((input_width / 32))
featmap_height=$((input_height / 32))

date=$(date +%Y%m%d%H%M)

# GPU configuration
device='0'
num_devices=$(echo $device | grep -o '[0-9]' | wc -l)
echo "num_devices=$num_devices"


# Select config file and annotation files based on task
if [ "$task" = "pretrain" ]; then
    config=cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_udp.py
    train_ann=annotations/train260807.json
    val_ann=annotations/val260807.json
elif [ "$task" = "pretrain_4kp" ]; then
    config=cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_4kp_10cls.py
    train_ann=annotations/train_filtered.json
    val_ann=annotations/val_filtered.json
elif [ "$task" = "pretrain_s" ]; then
    config=cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-s_udp.py
    train_ann=annotations/train.json
    val_ann=annotations/val.json
elif [ "$task" = "finetune" ]; then
    config=cfgs/rtmpose/finetune/rtmpose-m_${data_name}.py
    train_ann=annotations/train.json
    val_ann=annotations/val.json
elif [ "$task" = "pretrain_l" ]; then
    config=cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-l_8xb256-420e_coco-256x192-sy.py 
    train_ann=annotations/train260807.json
    val_ann=annotations/val260807.json
else
    echo "task=$task error, only support: pretrain, pretrain_4kp, pretrain_s, finetune"
    exit 1
fi

# Check config exists
if [ ! -f "$config" ]; then
    echo "Config file not found: $config"
    exit 1
fi

echo "Config: $config"
echo "Data: $data_name"
echo "Epochs: $set_epochs"
echo "Batch size: $batch_size"
echo "Image size: $img_size"
echo "Train ann: $train_ann"
echo "Val ann: $val_ann"

if [ -z "$resume" ]; then
    # Fresh training
    echo "Starting fresh training..."
    torchrun --nnodes=1 --nproc_per_node=$num_devices --master_port=40401 \
        tools/train.py $config \
        --work-dir ckpts/rtmpose/${data_name}/${date} \
        --amp \
        --auto-scale-lr \
        --cfg-options \
            train_dataloader.batch_size=$batch_size \
            train_dataloader.dataset.data_root=data/pose-dataset/${data_name} \
            train_dataloader.dataset.ann_file=$train_ann \
            train_dataloader.dataset.data_prefix.img=images \
            val_dataloader.batch_size=$((batch_size / 4)) \
            val_dataloader.dataset.data_root=data/pose-dataset/${data_name} \
            val_dataloader.dataset.ann_file=$val_ann \
            val_dataloader.dataset.data_prefix.img=images \
            val_evaluator.ann_file=data/pose-dataset/${data_name}/$val_ann \
            train_cfg.max_epochs=$set_epochs \
            codec.input_size=\(${input_width},\ ${input_height}\) \
            model.head.input_size=\(${input_width},\ ${input_height}\) \
            model.head.in_featuremap_size=\(${featmap_width},\ ${featmap_height}\) \
            model.head.decoder.input_size=\(${input_width},\ ${input_height}\) \
            train_dataloader.dataset.pipeline.5.input_size=\(${input_width},\ ${input_height}\) \
            train_dataloader.dataset.pipeline.8.encoder.input_size=\(${input_width},\ ${input_height}\) \
            val_dataloader.dataset.pipeline.2.input_size=\(${input_width},\ ${input_height}\) \
            train_pipeline.5.input_size=\(${input_width},\ ${input_height}\) \
            train_pipeline.8.encoder.input_size=\(${input_width},\ ${input_height}\) \
            val_pipeline.2.input_size=\(${input_width},\ ${input_height}\) 
else
    # Resume training
    date=$(echo $resume | cut -d'/' -f4 | cut -d'.' -f1)
    echo "Resuming from $resume, date=$date"
    torchrun --nnodes=1 --nproc_per_node=$num_devices --master_port=40401 \
        tools/train.py $config \
        --work-dir ckpts/rtmpose/${data_name}/${date} \
        --resume \
        --amp \
        --cfg-options \
            train_dataloader.batch_size=$batch_size \
            train_dataloader.dataset.data_root=data/pose-dataset/${data_name} \
            train_dataloader.dataset.ann_file=$train_ann \
            train_dataloader.dataset.data_prefix.img=images \
            val_dataloader.batch_size=$((batch_size / 4)) \
            val_dataloader.dataset.data_root=data/pose-dataset/${data_name} \
            val_dataloader.dataset.ann_file=$val_ann \
            val_dataloader.dataset.data_prefix.img=images \
            val_evaluator.ann_file=data/pose-dataset/${data_name}/$val_ann \
            train_cfg.max_epochs=$set_epochs \
            codec.input_size=\(${input_width},\ ${input_height}\) \
            model.head.input_size=\(${input_width},\ ${input_height}\) \
            model.head.in_featuremap_size=\(${featmap_width},\ ${featmap_height}\) \
            model.head.decoder.input_size=\(${input_width},\ ${input_height}\) \
            train_dataloader.dataset.pipeline.5.input_size=\(${input_width},\ ${input_height}\) \
            train_dataloader.dataset.pipeline.8.encoder.input_size=\(${input_width},\ ${input_height}\) \
            val_dataloader.dataset.pipeline.2.input_size=\(${input_width},\ ${input_height}\) \
            train_pipeline.5.input_size=\(${input_width},\ ${input_height}\) \
            train_pipeline.8.encoder.input_size=\(${input_width},\ ${input_height}\) \
            val_pipeline.2.input_size=\(${input_width},\ ${input_height}\)
fi