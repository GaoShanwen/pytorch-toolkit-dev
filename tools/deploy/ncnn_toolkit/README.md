### clone package from github

get newest version from this [link](https://github.com/Tencent/ncnn/releases/latest)

### build protobuf library (3.6.1)

### unzip package

these libraries are needed:

- **onnx2ncnn**: onnx -> ncnn (fp32)
- **ncnnoptimize**: ncnn -> ncnn (fp16)
- **ncnn2table**: quantize.txt -> table
- **ncnn2int8**: ncnn, table -> ncnn (fp32 -> int8)
- **ncnn2mem**: ncnn -> ncnn (encoded memory)
- **ncnnmerge**: merge ncnn

### Command Guides

```bash
<pytorch-toolkit-dev> ~$ # onnx optimize
    python -m onnxsim output/converted_model/mv4-2351.onnx output/converted_model/mv4-2351_sim.onnx
<pytorch-toolkit-dev> ~$ # onnx (optimized) -> ncnn (fp32)
    tools/deploy/ncnn_toolkit/onnx2ncnn output/converted_model/mv3-2066_sim.onnx output/converted_model/mv3-2066_sim.param \
        output/converted_model/mv3-2066_sim.bin
<pytorch-toolkit-dev> ~$ # ncnn (fp32) -> ncnn (fp16)
    tools/deploy/ncnn_toolkit/ncnnoptimize output/converted_model/mv3-2066_sim.param output/converted_model/mv3-2066_sim.bin \
        output/converted_model/mv3-2066_sim_fp16.param output/converted_model/mv3-2066_sim_fp16.bin 1
```
