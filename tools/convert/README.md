## Model Export

```bash
# Export model to ONNX format
~$ python tools/convert/export.py cfgs/rtmpose/pretrain_cspnext_udp/rtmpose-m_udp.py ckpts/rtmpose/BakingRefine/202608081607/epoch_40.pth

# Evaluate ONNX model
~$ python tools/convert/eval_onnx.py --ann data/pose-dataset/BakingRefine/annotations/val260807.json --onnx ckpts/rtmpose/BakingRefine/202608091326/best_coco_AP_epoch_90.onnx --input-size 256 192
```