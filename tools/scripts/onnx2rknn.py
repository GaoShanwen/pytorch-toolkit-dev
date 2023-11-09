######################################################
# author: gaowenjie
# email: gaowenjie@rongxwy.com
# date: 2023.11.09
# filenaem: onnx2rknn.py
# function: convert onnx model to rknn model.
######################################################
import argparse
import logging
import cv2
import os
import tqdm
import torch
import onnxruntime
import numpy as np
from rknn.api import RKNN

from timm.data import create_loader, resolve_data_config
from timm.models import create_model
from timm.utils import setup_default_logging, ParseKwargs

import sys
sys.path.append('./')

from local_lib.data import create_owner_dataset
from tools.feat_extract import init_feats_dir, merge_feat_files, save_feat

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('output', metavar='RKNN_FILE', help='output model filename')
parser.add_argument('--input', metavar='ONNX_FILE', help='output model filename')
parser.add_argument('--target-platform', '-tp', metavar='MODEL', default='rk3566',
                    help='target platform (default: rk3566)')
parser.add_argument('--do-quantizate', action='store_true', default=False,
                    help='enable do quantizate')
parser.add_argument('--results-dir', default='', type=str, metavar='FILEDIR',
                    help='Output feature file for validation results (summary)')
parser.add_argument('--data', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--infer-mode', default="val", type=str,
                    metavar='NAME', help='the dirs to inference.')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=2, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=3, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=[0.4850, 0.4560, 0.4060], metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=[0.2290, 0.2240, 0.2250], metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--num-choose', type=int, default=None,
                    help='Number choose in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cpu', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")


def evaliation(img_root, rknn, onnx_model, means=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]):
    top1_rknn, top1_onnx = 0, 0
    resize_width, resize_height = 256, 256
    # 指定裁剪后的图像大小  
    crop_height, crop_width = 224, 224
    img_num = 0

    ret = rknn.init_runtime()
    if ret != 0:
        print('init rknn failed!')
        exit(ret)
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(onnx_model, sess_options)
    input_name = session.get_inputs()[0].name

    for cat_index, cat in enumerate(sorted(os.listdir(img_root))):
        img_dir = os.path.join(img_root, cat)
        cat_rknn, cat_onnx = 0, 0
        for img_name in os.listdir(img_dir):
            img_path = os.path.join(img_dir, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_image = cv2.resize(img, (resize_width, resize_height), interpolation=cv2.INTER_CUBIC)
            
            # 计算裁剪的起始坐标  
            x = (resize_width - crop_width) // 2  
            y = (resize_height - crop_height) // 2  
            
            # 裁剪图像  
            cropped_image = resized_image[y:y+crop_height, x:x+crop_width, :]
            input = cropped_image[np.newaxis, ...]
            cropped_image = cropped_image/255.
            cropped_image = (cropped_image-means)/std
            # resized_image = cv2.resize(img, (480, 480))
            
            # # Inference
            # print('--> Running model')
            rknn_output = rknn.inference(inputs=[input.astype(np.float32)])
            output = session.run([], {input_name: [cropped_image.transpose((2,0,1))]})
            onnx_output = output[0]
            x = rknn_output[0]; output = np.exp(x)/np.sum(np.exp(x)); rknn_idx = np.argmax(output)
            x = onnx_output[0]; output = np.exp(x)/np.sum(np.exp(x)); onnx_idx = np.argmax(output)
            cat_rknn += rknn_idx == cat_index
            cat_onnx += onnx_idx == cat_index
            img_num += 1
        top1_rknn += cat_rknn
        top1_onnx += cat_onnx
        print(f"{cat}-onnx_idx: {cat_onnx/len(os.listdir(img_dir))} | rknn_idx: {cat_rknn/len(os.listdir(img_dir))}")
    print(f"onnx_top1: {top1_onnx}/{img_num}|{top1_onnx/img_num} | rknn_top1: {top1_rknn}/{img_num}|{top1_rknn/img_num}")


def feat_extract(args, rknn):
    root_dir = args.data or args.data_dir
    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]
    model = create_model(
        args.model, pretrained=args.pretrained, num_classes=args.num_classes,
        in_chans=in_chans, global_pool=args.gp, scriptable=args.torchscript,
        **args.model_kwargs,
    )
    
    def run_infer(model, args, root_dir, infer_mode, data_config):
        dataset = create_owner_dataset(
            root=root_dir, name=args.dataset, split=args.split,
            is_training=infer_mode=="train", download=args.dataset_download,
            load_bytes=args.tf_preprocessing, class_map=args.class_map,
            num_classes=args.num_classes,
            num_choose=args.num_choose,
        )
        _logger.info(f"load image number={len(dataset)}")
        device = torch.device(args.device)
        loader = create_loader(
            dataset, input_size=data_config['input_size'],
            batch_size=args.batch_size, is_training=False,
            use_prefetcher=not args.no_prefetcher,
            interpolation=data_config['interpolation'],
            mean=data_config['mean'], std=data_config['std'],
            num_workers=args.workers,
            crop_pct=data_config['crop_pct'],
            crop_mode=data_config['crop_mode'],
            pin_memory=args.pin_mem,
            device=device,
            tf_preprocessing=args.tf_preprocessing,
        )
        pbar = tqdm.tqdm(total=len(loader))

        init_feats_dir(args.results_dir)
        # sess_options = onnxruntime.SessionOptions()
        # sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        # session = onnxruntime.InferenceSession(args.input, sess_options)
        # input_name = session.get_inputs()[0].name
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            if args.do_quantizate:
                img = input.numpy().transpose((0,2,3,1))[0] * 255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                output = rknn.inference(inputs=[img[np.newaxis, ...]], data_format='nhwc')
                # output2 = session.run([], {input_name: [img.transpose((2,0,1))]})
            else:
                output = rknn.inference(inputs=[input.numpy()], data_format='nchw')
            # # output2 = session.run([], {input_name: input.numpy()})
            # import pdb; pdb.set_trace()
            save_feat(output[0], target, batch_idx, args.results_dir)
            pbar.update(1)

        # del pbar
        # merge_feat_files(args.results_dir, infer_mode)
        # _logger.info(f'feat saved in {args.results_dir}-{infer_mode}.npz')
    if args.do_quantizate:
        args.mean, args.std = [0,0,0], [1,1,1]
    data_config = resolve_data_config(
        vars(args), model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    run_infer(model, args, root_dir, 'val', data_config)
    run_infer(model, args, root_dir, 'train', data_config)


if __name__ == '__main__':
    setup_default_logging()
    args = parser.parse_args()
    # Create RKNN object
    rknn = RKNN(verbose=True)
    mean_vales = [num*255. for num in args.mean]
    std_values = [num*255. for num in args.std]
    rknn.config(mean_values=mean_vales, std_values=std_values, target_platform=args.target_platform) 
    print('Config model done')

    print('--> Loading model')
    onnx_model = args.input #"output/converted_model/20230925-185645-resnet18-224.onnx"
    ret = rknn.load_onnx(model=onnx_model)#, inputs=["input0"], input_size_list=[[1, args.in_chans, args.img_size, args.img_size]])
    if ret != 0:
        print('load model failed!')
        exit(ret)
    print('done')

    # Build model
    print('--> Building model')
    data_file='dataset/exp-data/minidata/quantizate/dataset1.txt' if args.do_quantizate else None
    ret = rknn.build(do_quantization=args.do_quantizate, dataset=data_file)#, rknn_batch_size=args.batch_size)
    # rknn.accuracy_analysis(inputs=['dataset/.../image.jpg'])
    if ret != 0:
        print('build model failed.')
        exit(ret)
    print('done')

    # Export rknn model
    print('--> Export RKNN model')
    ret = rknn.export_rknn(args.output)
    if ret != 0:
        print('Export rknn failed!')
        exit(ret)
    print('Export rknn success!')

    # # Set inputs
    # img_root = "dataset/exp-data/minidata/validation"
    # evaliation(img_root, rknn, onnx_model)
    ret = rknn.init_runtime()
    if ret != 0:
        print('init rknn failed!')
        exit(ret)
    # feat_extract(args, rknn)

    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(args.input, sess_options)
    input_name = session.get_inputs()[0].name
    img=cv2.imread("dataset/exp-data/minidata/quantizate/100_NZ53MZV0KS_1680344371005_1680344371719.jpg")
    inputs = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#; input = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    output = rknn.inference(inputs=[inputs[np.newaxis, ...]], data_format='nhwc')
    x = np.array(inputs).astype(np.float32)/255.  # ToTensor操作，将像素值范围从[0, 255]转换为[0.0, 1.0]  
    x = (x - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])  # Normalize操作，使用ImageNet标准进行标准化
    output2 = session.run([], {input_name: [x.transpose(2,0,1)]})
    print('rknn', output)
    print('onnx', output2)
    # import pdb; pdb.set_trace()
    rknn.release()
