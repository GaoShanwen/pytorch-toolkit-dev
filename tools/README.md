
## command

```bash
YOUR_DIR ~$ task=overflow;python3 inference.py --weights=../ckpts/$task.pt --interval=5 --json_port=12081 --rtsp_port=12082 --video=../dataset/$task.mp4 --conf-thres=0.6
```
