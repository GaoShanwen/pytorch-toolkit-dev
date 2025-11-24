# activate train envirment
docker run -it --gpus all --ipc host -v /home/gwj:/home/gwj \
-v /DATA1/gwj/py39:/root/anaconda3/envs/py39 -v /DATA1/gwj/.zshrc:/root/.zshrc \
-v /DATA1:/DATA1 -v /DATA2:/DATA2 -w $(pwd) \
--name GWJ003 nvidia/cuda:gwj