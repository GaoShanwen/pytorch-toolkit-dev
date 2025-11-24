docker run -e LD_LIBRARY_PATH=$PWD/tools/convert/lib:$LD_LIBRARY_PATH --network=host --rm --privileged -v $PWD:$PWD -it --gpus=all --name onnx-to-engine-yolov8 \
  -w $PWD yolov8-tensorrt-cpp:2.1 /bin/bash
export LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH 
# ./main <model.onnx>