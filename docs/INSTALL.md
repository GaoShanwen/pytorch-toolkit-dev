## install environment

reference [Ultralytics’ INSTALL](https://docs.ultralytics.com/zh/quickstart/) for details.

### Quick Start

```bash
<pytorch-toolkit-dev> ~$ # install ultralytics and its environment
<pytorch-toolkit-dev> ~$ git clone https://github.com/GaoShanwen/pytorch-toolkit-dev.git
<pytorch-toolkit-dev> ~$ apt install ffmpeg
<pytorch-toolkit-dev> ~$ git checkout yolo-dev
<pytorch-toolkit-dev> ~$ pip install -r docs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu102
<pytorch-toolkit-dev> ~$ # install opencv-contrib-python for yolov5 infer
<pytorch-toolkit-dev> ~$ pip install opencv-contrib-python
```

this command for `libgthread-2.0.so.0` error.

```bash
apt-get install libglib2.0-dev
```