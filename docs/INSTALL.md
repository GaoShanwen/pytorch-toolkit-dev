## 安装环境

### Install timm and Build this project

先安装python3.8对应环境，然后根据下面内容安装pytorch，timm等python库用于训练和推理任务。

```bash
<pytorch-toolkit-dev> ~$ # install timm and its environment(include rknn-tools2)
<pytorch-toolkit-dev> ~$ git clone https://github.com/GaoShanwen/pytorch-toolkit-dev.git
<pytorch-toolkit-dev> ~$ git checkout timm-dev
# <pytorch-toolkit-dev> ~$ pip install -r docs/requirements.txt # --extra-index-url https://download.pytorch.org/whl/cu117
<pytorch-toolkit-dev> ~$ pip install -r docs/requirements.txt --ignore-installed
<pytorch-toolkit-dev> ~$ python setup.py install
```
