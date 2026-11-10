### 环境安装问题

1.可以参考该[博客](https://blog.csdn.net/qq_42076902/article/details/129261266)解决如下问题：

```bash
    AttributeError: module ‘distutils‘ has no attribute ‘version‘
```

2.使用yum安装'mesa-libGL'解决如下问题：

```bash
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

3.加`HF_ENDPOINT=https://hf-mirror.com`到python前，解决预训练模型下载失败：

```bash
    # way1： add expoert HF_ENDPOINT=https://hf-mirror.com to ~/.bashrc
    export HF_ENDPOINT=https://hf-mirror.com
    HF_HUB_OFFLINE=1
    python3 -m transformers.cli download roberta-base
    # way2: add HF_ENDPOINT=https://hf-mirror.com to command line
    HF_ENDPOINT=https://hf-mirror.com python3 -m transformers.cli download roberta-base
```

4.重新安装编译make，gcc，glibc，解决如下问题：

```bash
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

5.安装解决‘nvcc --version’报错问题

```bash
    wget https://developer.download.nvidia.com/compute/cuda/12.2.0/local_installers/cuda_12.2.0_535.54.03_linux.run
    sh cuda_12.2.0_535.54.03_linux.run
    
    Do you accept the above EULA? (accept/decline/quit): 
    <$root> ~$ accept
    │ CUDA Installer                                                               │
    │ - [ ] Driver                                                                 │
    │      [ ] 535.54.03                                                           │
    │ + [X] CUDA Toolkit 12.2                                                      │
    │   [ ] CUDA Demo Suite 12.2                                                   │
    │   [ ] CUDA Documentation 12.2                                                │
    │ - [ ] Kernel Objects                                                         │
    │      [ ] nvidia-fs                                                           │
    │   Options                                                                    │
    │   Install 
    add this option to ~/.zshrc:
    export PATH=$PATH:$CUDA_HOME/bin
    export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

6.安装1.7.3版本的faiss-gpu，解决faiss-gpu卡住问题

```bash
    conda install -c pytorch -c nvidia faiss-gpu=1.7.3
```

7.解决只读文件系统导致的下载失败问题(unable to open file</root/.cache...> in read-only mode)

```bash
    chmod -R 777 /root/.cache/huggingface/hub/
```

8.安装protobuf==3.6.1后，建立软连接/lib64，解决调用ncnn是依赖项查找不到问题

```bash
    ImportError: libprotobuf.so.17: cannot open shared object file: No such file or directory
```