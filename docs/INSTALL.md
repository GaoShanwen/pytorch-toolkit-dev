## 安装环境

### Install timm

先安装python3.8对应环境，然后根据下面内容安装pytorch，timm等python库用于训练和推理任务。

```bash
<pytorch-toolkit-dev> ~$ # install timm and its environment(include rknn-tools2)
<pytorch-toolkit-dev> ~$ git clone https://github.com/GaoShanwen/pytorch-toolkit-dev.git
<pytorch-toolkit-dev> ~$ git checkout timm-dev
<pytorch-toolkit-dev> ~$ pip install -r docs/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu117
<pytorch-toolkit-dev> ~$ python setup.py install
```

```bash
<pytorch-toolkit-dev> ~$ yum install git tree curl vim zsh tmux -y # 安装git tree curl vim
<pytorch-toolkit-dev> ~$ # 生成github ssh key，添加到github账户中
<pytorch-toolkit-dev> ~$ git config --global user.name "your-name" # 设置用户名
<pytorch-toolkit-dev> ~$ git config --global user.email "your-email" # 设置邮箱
<pytorch-toolkit-dev> ~$ git config --global color.ui true # 开启git颜色显示
<pytorch-toolkit-dev> ~$ ssh-keygen -t rsa -b 4096 -C "your-email"
<pytorch-toolkit-dev> ~$ cat ~/.ssh/id_rsa.pub # 查看公钥
<pytorch-toolkit-dev> ~$ # 添加公钥到github账户中
<pytorch-toolkit-dev> ~$ ssh -T git@github.com # 测试debug是否成功
<pytorch-toolkit-dev> ~$ chsh -s /bin/zsh # 更换zsh为默认shell
<pytorch-toolkit-dev> ~$ # 安装zsh插件
<pytorch-toolkit-dev> ~$ # clone oh-my-zsh
<pytorch-toolkit-dev> ~$ git clone https://gitee.com/mirrors/oh-my-zsh.git ~/.oh-my-zsh
<pytorch-toolkit-dev> ~$ git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/plugins/zsh-autosuggestions
<pytorch-toolkit-dev> ~$ git clone https://github.com/zsh-users/zsh-syntax-highlighting ~/.oh-my-zsh/plugins/zsh-syntax-highlighting
<pytorch-toolkit-dev> ~$ # clone autojump
<pytorch-toolkit-dev> ~$ git clone https://gitee.com/gentlecp/autojump.git; cd autojump; ./install.sh;  # 加自动跳转
<pytorch-toolkit-dev> ~$ vim ~/.zshrc # 编辑zsh配置文件
plugins=(git zsh-autosuggestions zsh-syntax-highlighting) # 启用插件
[[ -s /root/.autojump/etc/profile.d/autojump.sh ]] && source /root/.autojump/etc/profile.d/autojump.sh
<pytorch-toolkit-dev> ~$ source ~/.zshrc # 使配置文件生效
# Retach userspaces
<pytorch-toolkit-dev> ~$ set -g default-command "reattach-to-user-namespace -l zsh"
```

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

### 环境安装问题

1.可以参考该[博客](https://blog.csdn.net/qq_42076902/article/details/129261266)解决如下问题：

```bash
    AttributeError: module ‘distutils‘ has no attribute ‘version‘
```

2.使用yum安装'mesa-libGL'解决如下问题：

```bash
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

3.加‘HF_ENDPOINT=https://hf-mirror.com’到python前，解决预训练模型下载失败：

···bash
    # way1： add expoert HF_ENDPOINT=https://hf-mirror.com to ~/.bashrc
    export HF_ENDPOINT=https://hf-mirror.com
    python3 -m transformers.cli download roberta-base
    # way2: add HF_ENDPOINT=https://hf-mirror.com to command line
    HF_ENDPOINT=https://hf-mirror.com python3 -m transformers.cli download roberta-base
···

4.重新安装编译make，gcc，glibc，解决如下问题：

```bash
    ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```