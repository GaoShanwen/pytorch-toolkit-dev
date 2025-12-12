## convert2onnx

```bash
    ~$ for yolov8-detect
    ~$ # add library
    ~$ ln -s /root/anaconda3/envs/py39/lib/python3.9/site-packages/nvidia/cublas/lib/libcublas*.so.11 /usr/lib/
    ~$ # convert2onnx
    ~$ python tools/convert/export.py -t vehicle -f onnx
    ~$ # convert2engine
    ~$ python tools/convert/export.py -t vehicle -f engine
```

## convert2trt

```bash
    ~$ sh tools/convert/convert2trt.sh
    ~$ tools/convert/main tools/convert/model/vehicle_202511062316.onnx
```
