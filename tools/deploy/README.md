## Command Guides

<details open>

<summary>Convert Pth Model To ONNX/RKNN</summary>

- **Pth->ONNX**

```bash
<pytorch-toolkit-dev> ~$ # pth -> onnx
        python tools/deploy/onnx_export.py output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx \
        -m mobilenetv3_redution_large_100 --img-size 224 --opset 12 --num-classes 4281 \
        --checkpoint output/train/20231022-213614-mobilenetv3_redution_large_100-224/model_best.pth.tar 
```

- **ONNX->RKNN**

```bash
<pytorch-toolkit-dev> ~$ # onnx -> rknn; validate(onnx and rknn, if model is cls model) model
        python tools/deploy/onnx2rknn.py output/converted_model/rk3566-mobilenetv3-224.rknn \
        --input output/converted_model/20231022-213614-mobilenetv3_redution_large_100-224.onnx
<pytorch-toolkit-dev> ~$ # convert other model
        ... output/converted_model/rk3566-regnety_016-224.rknn --input output/converted_model/... 
```

</details>
