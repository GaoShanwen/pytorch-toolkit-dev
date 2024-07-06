
#!/bin/bash
task_root=dataset/feature_pack
rmodel_dir=$task_root/rmodel
task_dir=$task_root/tasklist
txt_dir=$task_root/tmp/imglist

while true; 
do
    task_num=$(find dataset/feature_pack/tasklist/*.txt -type f | wc -l)
    if [ $task_num -eq 0 ]; then
        echo "No task found, waiting for 20 seconds..."
        sleep 20000
    else
        for task_file in $task_dir/*.txt;
        do
            brand_id=$(basename $task_file | cut -d'.' -f1)
            echo "starting process for brand_id=$brand_id..."
            # create txt file for readable images in this brand.
            mkdir $txt_dir/$brand_id
            python tools/task/create_readable_dataset.py $task_root/gallery/$brand_id $txt_dir/$brand_id/train.txt
            echo Total images in brand=$brand_id is: $(cat $txt_dir/$brand_id/train.txt | wc -l)
            for rmodel_version in `ls $rmodel_dir`;
            do
                # 获取模型及类别数
                num_classes=$(echo $rmodel_version | cut -d_ -f2-)
                version_num=$(echo $rmodel_version | cut -d_ -f1)
                # 获取空闲显卡, 每20秒再查询一次，直到有空闲显卡
                check_free_result=$(python tools/task/get_free_gpu.py)
                # 有空闲卡则开始转特征库
                if [[ $check_free_result == *"true"* ]]; then
                    gpu_id=${check_free_result: -1}
                    echo "Starting process $process_i on GPU $gpu_id"
                    # convert imgs to feats
                    CUDA_VISIBLE_DEVICES=$gpu_id python tools/post/feat_extract.py --config cfgs/base-regnety_redution_040.ra3_in1k.yaml --options \
                    batch_size=1024 model_classes=$num_classes num_gpu=1 checkpoint=$rmodel_dir/$rmodel_version/model.pth.tar cats_path='' \
                    results_dir=$task_root/tmp/feats/r${version_num}_$brand_id data_dir=$txt_dir/$brand_id infer_mode=train
                    # convert npz to bin
                    mkdir -p $task_root/nx/$brand_id/R${version_num}
                    python tools/task/convert_npz2bin.py $task_root/tmp/feats/r${version_num}_$brand_id-train.npz \
                        $task_root/tmp/bin/r${version_num}_$brand_id.bin --src-img-dir $task_root/gallery/$brand_id \
                        --label-file $task_root/nx/$brand_id/R${version_num}/labelext.txt --brand-id $brand_id
                    # aes ecb encode the bin file
                    ./tools/task/write_nx $task_root/tmp/bin/r${version_num}_$brand_id.bin $task_root/nx/$brand_id/R${version_num}/modelnew.nx
                    version=$(echo $rmodel_version | cut -d- -f1)
                    cp $rmodel_dir/$rmodel_version/r$version.zip $task_root/nx/$brand_id/R${version_num}/
                else
                    echo "error", $check_free_result
                fi
            done
        done
    fi
done


