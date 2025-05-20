dir=$1
sh run_docker.sh
python data/scripts/mp_video2img.py -v $dir/videos -i $dir/images # -g 400
python data/scripts/mp_rm_similarities.py -i $dir/images -d $dir/unlabeled