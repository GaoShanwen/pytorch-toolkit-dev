
## command for yolov8-pose-train

```bash
<pytorch-toolkit-dev> ~$ sh tools/train.sh trainval_set '' 100 16 640 pose
```

## command for yolov8-pose-infer

```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --weights <runs/...best.pt> --img_path <img_path> --options symmetry_match=True
```


