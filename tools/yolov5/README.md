
## command for yolov5-det

```bash
YOUR_DIR ~$ cd tools/yolov5
YOUR_DIR ~$ task=partmissing;python3 v5_inference.py --weights=../../ckpts/$task.pt --interval=1 --json_port=12081 --rtsp_port=12082 --source=../../data/test_videos/$task.mp4 --half --conf-thres=0.3 --line-thickness=1 --data=$task.yaml
```
