######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.05
# filenaem: convert server for convert_imgs2bin task.
######################################################
#!/bin/bash
rmodel_version=$1
gpu_id=$2
rmodel_dir=$3
txt_dir=$4
task_root=$5
brand_id=$6
used_dir=$7
batch_size=$8
# 获取模型及类别数
num_classes=$(echo $rmodel_version | cut -d_ -f2-)
version_num=$(echo $rmodel_version | cut -d_ -f1)
# convert imgs to feats
CUDA_VISIBLE_DEVICES=$gpu_id python tools/post/feat_extract.py --config cfgs/base-regnety_redution_040.ra3_in1k.yaml --options \
    batch_size=$batch_size model_classes=$num_classes num_gpu=1 checkpoint=$rmodel_dir/$rmodel_version/model.pth.tar cats_path='' \
    results_dir=$task_root/tmp/feats/r${version_num}_$brand_id data_dir=$txt_dir/$brand_id infer_mode=train tqdm_disabled=True
# free gpu flag when finished features extraction.
rm -f $used_dir/$gpu_id
# convert npz to bin
mkdir -p $task_root/nx/$brand_id/R${version_num}
python project/convert_img2bin/convert_npz2bin.py $task_root/tmp/feats/r${version_num}_$brand_id-train.npz \
    $task_root/tmp/bin/r${version_num}_$brand_id.bin --src-img-dir $task_root/gallery/$brand_id \
    --label-file $task_root/nx/$brand_id/R${version_num}/labelext.txt --brand-id $brand_id
# aes ecb encode the bin file
./project/convert_img2bin/write_nx.so $task_root/tmp/bin/r${version_num}_$brand_id.bin $task_root/nx/$brand_id/R${version_num}/modelnew.nx
version=$(echo $rmodel_version | cut -d- -f1)
cp $rmodel_dir/$rmodel_version/*.zip $task_root/nx/$brand_id/R${version_num}/
cp $rmodel_dir/$rmodel_version/*.onnx $task_root/nx/$brand_id/R${version_num}/
# remove tmp directory
rm -rf $task_root/tmp/feats/r${version_num}_$brand_id-train.npz
rm -rf $task_root/tmp/bin/r${version_num}_$brand_id.bin
# rm -rf $task_root/tasklist/$brand_id.txt
# # publish
# ossutil cp -r -u $task_root/nx/$brand_id/R${version_num}/labelext.txt oss://rx-gallery/$brand_id/rmodelnx/
# ossutil cp -r -u $task_root/nx/$brand_id/R${version_num}/modelnew.nx oss://rx-gallery/$brand_id/rmodelnx/
# ossutil cp -r -u $task_root/nx/$brand_id/R${version_num}/*.zip oss://rx-gallery/$brand_id/rbig/
# ossutil cp -r -u $task_root/nx/$brand_id/R${version_num}/*.onnx oss://rx-gallery/$brand_id/onnx/

# ossutil cp -r -u dataset/feature_pack/nx/1386/Rv4.2-240726/labelext.txt oss://rx-gallery/1482/rmodelnx/
# ossutil cp -r -u dataset/feature_pack/nx/1386/Rv4.2-240726/modelnew.nx oss://rx-gallery/1482/rmodelnx/
# ossutil cp -r -u dataset/feature_pack/nx/1386/Rv4.2-240726/*.zip oss://rx-gallery/1386/rbig/
# ossutil cp -r -u dataset/feature_pack/nx/1386/Rv4.2-240726/*.onnx oss://rx-gallery/1386/onnx/
