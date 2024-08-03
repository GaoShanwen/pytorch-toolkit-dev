
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

while true; 
do
    start_time=$(date "+%Y-%m-%d %H:%M:%S")
    task_num=$(find dataset/feature_pack/tasklist/*.txt -type f | wc -l)
    if [ $task_num -eq 0 ]; then
        echo "No task found, waiting for 20 seconds..."
        sleep 20
    else
        total_img_count=0
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
            printf "\033[0m\n[\033[32m%4s\033[0m /\033[32m%4s\033[0m]" "$this_task_count" "$task_num" 
            printf "\033[34m Task dataset info \033[0m|"
            printf "\033[34m brand_id:\033[31m%5d\033[34m, images:\033[33m%6d\033[34m," "$brand_id" "$this_img_count"
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
                    printf " Running task on GPU \033[1;35m$gpu_id\033[0m |"
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
    printf "\033[0m\nstart-time: \033[1;36m%19s" "$start_time"
    printf "\033[0m, end-time: \033[1;36m%19s" "$(date "+%Y-%m-%d %H:%M:%S")"
    echo $(printf "\033[0m, total img count: \033[1;33m$total_img_count")
    break # will be removed after testing
done
