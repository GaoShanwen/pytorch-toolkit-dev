## YOLOv8 Pose Toolkit

### Project Structure

```
tools/
├── predict.py          # Inference script
├── train.py            # Training script
├── train.sh            # Training launch script
├── val.py              # Validation script
├── stream.py           # Stream inference script
├── inference.py        # Batch inference script
└── convert/
    └── export.py       # Model export script
```

### Dataset Structure Requirements

Training datasets must be placed under `data/pose-dataset/` directory:

```bash
 $<pytorch-toolkit-dev> ~$ python3 data/scripts/pose/preprocess_coco.py --mode yolo2coco --dataset_dir data/pose-dataset/BakingRecognize --train_file train260807.txt --out_file data/pose-dataset/BakingRefine/annotations/train260807.json
```

```
data/pose-dataset/
├── <dataset_name1>/           # Dataset name (e.g., trainval_set)
│   ├── dataset.yaml          # Dataset configuration file
│   ├── images/
│   ├── train.txt            # Training images list
│   ├── val.txt              # Validation images list
│   └── labels/
└── <dataset_name2>/
```

**Note**: The dataset name must match the folder name under `data/pose-dataset/`.

### Training Commands

```bash
<pytorch-toolkit-dev> ~$ sh tools/train.sh <dataset_name> [resume] [epochs] [batch_size] [img_size] [task]
```

**Parameter Description**:
- `dataset_name`: Dataset name (must exist under `data/pose-dataset/`)
- `resume`: Optional, path to checkpoint for resuming training, empty string for training from scratch
- `epochs`: Optional, number of epochs, default 100
- `batch_size`: Optional, batch size, default 16
- `img_size`: Optional, image size, default 640
- `task`: Optional, task type, supports detect/segment/pose, default detect

**Examples**:
```bash
# Train pose estimation model
<pytorch-toolkit-dev> ~$ sh tools/train.sh BakingRefine '' 100 64 192 pretrain

# Resume training from checkpoint
<pytorch-toolkit-dev> ~$ sh tools/train.sh trainval_set runs/pose/ckpts/trainval_set/202607211944/weights/best.pt 50 16 640 pose
```

### Validation Commands

```bash
<pytorch-toolkit-dev> ~$ file=annotations/val260817e.json;data_root=data/pose-dataset/BakingRefine;python3 tools/val.py ckpts/rtmpose/BakingRefine/202608172328/rtmpose-l_8xb64-100e_260817-192x192-sy.py ckpts/rtmpose/BakingRefine/202608172328/best_coco_AP_epoch_70.pth --show-dir vis --cfg-options default_hooks.visualization.enable=True visualizer.type=DynamicPoseVisualizer val_dataloader.dataset.data_root=$data_root val_dataloader.dataset.ann_file=$file \
    test_dataloader.dataset.data_root=$data_root test_dataloader.dataset.ann_file=$file \
    val_evaluator.ann_file=$data_root/$file test_evaluator.ann_file=$data_root/$file \
    test_dataloader.dataset.pipeline.2.input_size=192,192 visualizer.save_by_category=True
```

### Inference Commands

```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_udp.py \
                            ckpts/rtmpose/BakingRefine/202608081607/epoch_40.pth <图片路径> --out-file <输出路径>
```

**Parameter Description**:
- `--weights`: Path to model weights
- `--img_path`: Path to input image/video
- `--flip`: Optional, enable horizontal flip inference augmentation
- `--save`: Optional, save inference results

**Examples**:
```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --weights runs/pose/ckpts/trainval_set/202607211944/weights/best.pt --img_path data/pose-dataset/demo.jpeg --flip --save
```

### Model Export Commands

```bash
<pytorch-toolkit-dev> ~$ # Export model to ONNX format
<pytorch-toolkit-dev> ~$ python3 tools/convert/export.py cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_udp.py \
                            ckpts/rtmpose/BakingRefine/202608081607/epoch_40.pth
<pytorch-toolkit-dev> ~$ # Evaluate ONNX model
<pytorch-toolkit-dev> ~$ python tools/convert/eval_onnx.py --ann data/pose-dataset/BakingRefine/annotations/val260807.json --onnx ckpts/rtmpose/BakingRefine/202608091326/best_coco_AP_epoch_90.onnx --input-size 256 192
```

