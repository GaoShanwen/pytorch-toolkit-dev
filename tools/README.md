## RFDETR Toolkit

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

Training datasets must be placed under `data/det-dataset/` directory:

```
data/det-dataset/
├── <dataset_name1>/           # Dataset name (e.g., trainval_set)
│   ├── dataset.yaml          # Dataset configuration file
│   ├── images/
│   ├── train.txt            # Training images list
│   ├── val.txt              # Validation images list
│   └── labels/
└── <dataset_name2>/
```

**Note**: The dataset name must match the folder name under `data/det-dataset/`.

### Training Commands

```bash
<pytorch-toolkit-dev> ~$ sh tools/train.sh <dataset_name> [resume] [epochs] [batch_size] [img_size] [task]
```

**Parameter Description**:
- `dataset_name`: Dataset name (must exist under `data/det-dataset/`)
- `resume`: Optional, path to checkpoint for resuming training, empty string for training from scratch
- `epochs`: Optional, number of epochs, default 100
- `batch_size`: Optional, batch size, default 16
- `img_size`: Optional, image size, default 640
- `task`: Optional, task type, supports detect/segment/pose, default detect

**Examples**:
```bash
# Train detection model
<pytorch-toolkit-dev> ~$ sh tools/train.sh BakingRecognizeCOCO '' 100 16 384 detect nano

# Resume training from checkpoint
<pytorch-toolkit-dev> ~$ sh tools/train.sh BakingRecognizeCOCO ckpts/detect/BakingRecognizeCOCO/202608180017/checkpoint_best_total.pth 100 8 384 nano detect
```

### Validation Commands

```bash
<pytorch-toolkit-dev> ~$ python3 tools/val.py --weights <runs/.../best.pt> --data <dataset_name> [--symmetry_match] [--mixed_data]
```

### Inference Commands

```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --img-path <img_path> --model <model_path>
```

**Examples**:
```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --img-path /home/wenjie/Downloads/detect --model ckpts/detect/BakingRecognizeCOCO/202608180017/checkpoint_best_total.pth
```