# TensorRT conversion for pose models
docker run -e LD_LIBRARY_PATH=$PWD/tools/convert/lib:$LD_LIBRARY_PATH --network=host --rm --privileged -v $PWD:$PWD -it --gpus=all --name onnx-to-engine-pose \
  -w $PWD pose-tensorrt-cpp:1.0 /bin/bash
export LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH
# ./main <model.onnx>