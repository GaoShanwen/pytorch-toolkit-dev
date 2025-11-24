
## command for yolov8-seg-train

```bash
<pytorch-toolkit-dev> ~$ sh tools/train.sh overflow yolov8n-seg.pt '' 20 64 640 seg
```

## command for yolov8-seg-infer

```bash
<pytorch-toolkit-dev> ~$ task=overflow;python3 tools/inference.py --weights=ckpts/$task.pt --interval=5 --json_port=12081 --rtsp_port=12082 --video=data/test_videos/$task.mp4 --conf-thres=0.6
```

```bash
<pytorch-toolkit-dev> ~$ python tools/val.py --data data/det-dataset/vehicle/ultralytics.yaml --task detect --option save_txt=true save_conf=true with_ignore=true --device 0,1,2,3,4,5,6,7 --batch 256 --model <last.pt>
```

```bash
<pytorch-toolkit-dev> ~$ python data/scripts/detect/vis_badcase.py -t vehicle -s data/det-dataset/vehicle/v_val.txt
```
