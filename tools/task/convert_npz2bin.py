######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2024.07.05
# filenaem: convert_npz2bin.py
# function: create readable dataset from raw dataset.
######################################################
import numpy as np
# import struct
import os
import json
import faiss
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Convert npz to bin')
    parser.add_argument('src_file', type=str, help='source file path')
    parser.add_argument('dst_file', type=str, help='destination file path')
    parser.add_argument('--src-img-dir', type=str, help='src directory of images')
    parser.add_argument('--label-file', type=str, help='label file path')
    parser.add_argument('--prefix', type=str, default='dataset/feature_pack/gallery', help='prefix of file path')
    parser.add_argument('--brand-id', type=str, help='brand id')
    return parser.parse_args()


def run_convert(src_file, dst_file, index_map, label_file, prefix, brand_id=None):
    data = np.load(src_file)
    feats, labels, fpaths = data['feats'].astype(np.float32), data['gts'], data['fpaths']
    faiss.normalize_L2(feats)

    label_list = sorted(list(set(labels)))
    label_map = {i: index_map[label] for i, label in enumerate(label_list)}
    with open(label_file, 'w', encoding='utf-8') as f:
        json.dump(label_map, f)
    labels = np.where(labels[:, None] == label_list)[1] # start from 0 for labels in labels list
    
    real_size = min(120000, labels.shape[0])
    dst_feats, dst_labels = np.zeros((120000, 128), dtype=np.float32), np.zeros((120000), dtype=np.int32)
    dst_feats[:real_size, :], dst_labels[:real_size] = feats, labels
    zero_array = np.zeros((120000,), dtype=np.int32)

    dst_fpaths = np.array([["",]]*120000, dtype='S150').flatten()
    dst_fpaths[:real_size] = np.array([path.replace(f"{prefix}/{brand_id}", f"{brand_id}/backflow") for path in fpaths])
    
    int_array = np.array([120000, 128, 120000, real_size]+[0,]*6, dtype=np.int32)
    with open(dst_file, 'wb') as f:
        int_array.tofile(f) # 40
        dst_feats.flatten().tofile(f) # 4*128*120000
        dst_labels.tofile(f) # 4*120000
        for i in range(4): # 4*4*120000
            zero_array.tofile(f)
        dst_fpaths.tofile(f)


if __name__ == '__main__':
    args = parse_args()
    index_map = {int(label): label for label in os.listdir(args.src_img_dir) if label.isdecimal()}
    run_convert(args.src_file, args.dst_file, index_map, args.label_file, args.prefix, args.brand_id)


        