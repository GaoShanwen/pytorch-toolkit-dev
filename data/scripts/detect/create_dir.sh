data_dir=$1 #data/det-dataset/candp/shanxixialiyuan/*/*
obj_path=$2 #data/det-dataset/candp/expend.txt
find $data_dir -type f | grep '.jpg' > $obj_path
