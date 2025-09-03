docker run -e LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH --network=host --rm --privileged -v $PWD:/home -it --gpus=all --name onnx-to-engine-yolov8 \
  -w /home yolov8_tensorrt_cpp:2.1 /bin/bash
export LD_LIBRARY_PATH=/home/lib:$LD_LIBRARY_PATH 
# ./main <model.onnx>