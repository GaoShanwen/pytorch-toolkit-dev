
#!/bin/bash
task_root=dataset/feature_pack
rmodel_dir=$task_root/rmodel
task_dir=$task_root/tasklist
txt_dir=$task_root/tmp/imglist
log_dir=$task_root/tmp/log
used_dir=$task_root/tmp/used_gpus
start_gpu_id=0
end_gpu_id=8
batch_size=1024

red='\033[0;31m'
green='\033[0;32m'
yellow='\033[0;33m'
blue='\033[0;34m'
magenta='\033[0;35m'
cyan='\033[0;36m'
white='\033[0;37m'

RED='\033[1;31m'
GREEN='\033[1;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
MAGENTA='\033[1;35m'
CYAN='\033[1;36m'
WHITE='\033[1;37m'

RESET='\033[0m'

while true; 
do
    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    task_num=$(find dataset/feature_pack/tasklist/*.txt -type f | wc -l)
    model_num=$(ls $rmodel_dir | wc -l)
    total_img_count=0
    if [ $task_num -eq 0 ]; then
        echo -e "${red}No task found, waiting for 20 seconds...${RESET}"
        sleep 20
    elif [ $model_num -eq 0 ]; then
        echo -e "${red}No model found, waiting for 20 seconds...${RESET}"
        sleep 20
    else
        this_task_count=1
        printf "Starting convert imgs to bin for $task_num tasks..."
        rm $used_dir/* 2>/dev/null
        for task_file in $task_dir/*.txt;
        do
            brand_id=$(basename $task_file | cut -d'.' -f1)
            # create txt file for readable images in this brandd.
            mkdir $txt_dir/$brand_id 2>/dev/null
            python project/convert_img2bin/create_readable_dataset.py $task_root/gallery/$brand_id $txt_dir/$brand_id/train.txt
            this_img_count=$(cat $txt_dir/$brand_id/train.txt | wc -l)
            # display the dataset information.
            printf "${RESET}\n[${green}%4s${RESET} /${green}%4s${RESET}]" "$this_task_count" "$task_num" 
            printf "${blue} Task dataset info ${RESET}|"
            printf "${blue} brand_id:${red}%5d${blue}, images:${yellow}%6d${blue}," "$brand_id" "$this_img_count"
            total_img_count=$(expr $total_img_count + $this_img_count)
            this_task_count=$(expr $this_task_count + 1)
            # 开始转特征库
            for rmodel_version in `ls $rmodel_dir`;
            do
                # 获取空闲显卡, 每20秒再查询一次，直到有空闲显卡
                free_result=$(python project/convert_img2bin/get_free_gpu.py -s $start_gpu_id -e $end_gpu_id -t 10 -r $used_dir)
                # 有空闲卡则开始转特征库
                if [[ $free_result == *"true"* ]]; then
                    gpu_id=${free_result: -1}
                    printf " Running task on GPU ${magenta}$gpu_id${RESET} |"
                    nohup sh ./project/convert_img2bin/convert_server.sh $rmodel_version $gpu_id $rmodel_dir $txt_dir \
                            $task_root $brand_id $used_dir $batch_size > $log_dir/$brand_id-$rmodel_version.log 2>&1 &
                else
                    echo "error", $free_result
                fi
            done
        done
    fi
    while [ $(find $used_dir -type f | wc -l) -gt 0 ]; do
        sleep 5
    done
    printf "${RESET}\nstart-time: ${CYAN}%19s" "$start_time"
    printf "${RESET}, end-time: ${CYAN}%19s" "$(date "+%Y-%m-%d %H:%M:%S")"
    printf "${RESET}, total img count: ${YELLOW}%d${RESET}\n" "$total_img_count"
    break # will be removed after testing
done