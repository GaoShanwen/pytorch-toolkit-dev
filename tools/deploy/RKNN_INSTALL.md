### RKNN 环境准备

如果需要转换模型，则需另外安装以下内容：

+ [make=4.2](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md#安装-make-42)
+ [gcc=8.2](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md#安装gcc820)
+ [glib=2.29](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md#安装glibc-229)
+ [rknn-tools2 for py38](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md#安装rknn)
+ [apex=0.1](https://github.com/GaoShanwen/pytorch-toolkit-dev/blob/timm-dev/docs/environment.md#apex安装)

### 安装gcc和make依赖，先安装低版本用于编译环境

```bash
    yum install bison wget bzip2 gcc gcc-c++ glib-headers make -y
```

### 安装 make-4.2

```bash
    cd <your-workspace>
    wget http://ftp.gnu.org/gnu/make/make-4.2.1.tar.gz
    tar -zxvf make-4.2.1.tar.gz; cd make-4.2.1
    mkdir build; cd build
    ../configure --prefix=/usr/local/make && make && make install
    export PATH=/usr/local/make/bin:$PATH
    ln -s /usr/local/make/bin/make /usr/local/make/bin/gmake
    make -v # make sure make-4.2 installed successfully
```

### 安装gcc8.2.0

```bash
    cd <your-workspace>
    wget https://mirrors.tuna.tsinghua.edu.cn/gnu/gcc/gcc-8.2.0/gcc-8.2.0.tar.gz
    tar xf gcc-8.2.0.tar.gz && cd gcc-8.2.0
    # 下载gmp mpfr mpc等供编译需求的依赖项
    ./contrib/download_prerequisites
    # 配置
    mkdir build && cd build
    ../configure --prefix=/usr/local/gcc-8.2.0 --enable-bootstrap --enable-checking=release --enable-languages=c,c++ --disable-multilib
    # 编译安装(该命令持续很久（>1h），先启动后台再运行)
    make -j 4 && make install 
    # 修改环境变量，使得gcc-8.2.0为默认的gcc
    # 将 "export PATH=/usr/local/gcc-8.2.0/bin:$PATH" 加到~/.zshrc文件
    vim ~/.zshrc # 编辑zsh配置文件
export PATH=/usr/local/gcc-8.2.0/bin:$PATH
    source ~/.zshrc # 使配置文件生效
    # 加软连接
    ln -sv /usr/local/gcc-8.2.0/include/ /usr/include/gcc
"/usr/include/gcc/include" -> "/usr/local/gcc-8.2.0/include/"
    gcc -v # make sure gcc-8.2.0 installed successfully
```

### 安装glibc-2.29

```bash
    cd <your-workspace>
    wget https://ftp.gnu.org/gnu/glibc/glibc-2.29.tar.gz
    tar -xvf glibc-2.29.tar.gz; cd glibc-2.29
    # 创建build目录并进入build目录
    mkdir build; cd build
    # 执行./configure
    ../configure --prefix=/usr --disable-profile --enable-add-ons --with-headers=/usr/include --with-binutils=/usr/bin
    # 编译安装(该命令持续很久，先启动后台再运行, 执行时可能出现错误，可以忽略)
    make && make install
    ## 查看共享库
    ls -l /lib64/libc.so.6
    ## 再次查看系统中可使用的glibc版本
    strings /lib64/libc.so.6 |grep GLIBC_
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/glibc-2.29/lib
    LD_PRELOAD=/lib64/libc-2.28.so; rm -rf /lib64/libc.so.6
    LD_PRELOAD=/lib64/libc-2.28.so; ln -s /lib64/libc-2.28.so /lib64/libc.so.6
    ldconfig # 更新状态
```

### 安装rknn

```bash
    git clone https://github.com/rockchip-linux/rknn-toolkit2.git
    cd rknn-toolkit2; pip install rknn-toolkit2/packages/rknn_toolkit2-*-cp38-cp38-linux_x86_64.whl --no-deps
```

### 安装apex

```bash
    git clone -b master https://gitee.com/ascend/apex.git && cd apex/
    bash scripts/build.sh --python=3.8
    pip install -r requirements.txt --no-deps -t p/root/anaconda3/envs/py38/lib/python3.8/site-packages
    pip3 uninstall apex
    pip3 install --upgrade apex-0.1+ascend-{version}.whl
    pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

    <pytorch-toolkit-dev> ~$ # install apex for speed training
    < other-workspace > ~$ git clone https://github.com/NVIDIA/apex
    < other-workspace > ~$ cd apex
    < other-workspace/apex > ~$ pip install -v --no-cache-dir --no-build-isolation --global-option="--cpp_ext" --global-option="--cuda_ext" --global-option="--deprecated_fused_adam" --global-option="--xentropy" --global-option="--fast_multihead_attn" ./
```
