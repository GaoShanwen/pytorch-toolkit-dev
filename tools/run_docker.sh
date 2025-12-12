# apt install zsh # if docker run failure due of 'zsh'
# activate train envirment
docker run --ipc=host -it -p 12081:12081 -p 12082:12082 --name GWJ003 \
--gpus all --cap-add sys_ptrace --privileged -v /home/gwj:/home/gwj \
-v /Dataset/VehicleWIthPersonandTraffic:/Dataset/VehicleWIthPersonandTraffic \
-v /mnt/bak/gwj:/mnt/bak/gwj -w $(pwd) --cap-add sys_ptrace nvidia/cuda-py39:gwjv1.0 zsh