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
├── convert/
│   └── export.py       # Model export script
└── yolov5/             # YOLOv5 related tools
```

### Dataset Structure Requirements

Training datasets must be placed under `data/pose-dataset/` directory:

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
<pytorch-toolkit-dev> ~$ sh tools/train.sh trainval_set '' 100 16 640 pose

# Resume training from checkpoint
<pytorch-toolkit-dev> ~$ sh tools/train.sh trainval_set runs/pose/ckpts/trainval_set/202607211944/weights/best.pt 50 16 640 pose
```

### Inference Commands

```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --weights <runs/.../best.pt> --img_path <img_path> [--flip] [--save]
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