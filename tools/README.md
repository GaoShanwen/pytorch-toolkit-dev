
## command for yolov8-pose-train

```bash
<pytorch-toolkit-dev> ~$ sh tools/train.sh trainval_set '' 100 16 640 pose
```

## command for yolov8-pose-infer

```bash
<pytorch-toolkit-dev> ~$ python3 tools/predict.py --weights=runs/pode/ckpts/---best.pt --img_path=data/test_videos/ --options symmetry_match=True
```

```bash
<pytorch-toolkit-dev> ~$ python tools/val.py --data data/det-dataset/vehicle/ultralytics.yaml --task detect --option save_txt=true save_conf=true with_ignore=true --device 0,1,2,3,4,5,6,7 --batch 256 --model <last.pt>
```

```bash
<pytorch-toolkit-dev> ~$ python data/scripts/detect/vis_badcase.py -t vehicle -s data/det-dataset/vehicle/v_val.txt
```
