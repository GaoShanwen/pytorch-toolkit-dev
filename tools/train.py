import sys
import argparse
from rfdetr import (
    RFDETRNano, RFDETRSmall, RFDETRMedium, RFDETRLarge,
    RFDETRSegNano, RFDETRSegSmall, RFDETRSegMedium, RFDETRSegLarge,
    RFDETRSegXLarge, RFDETRSeg2XLarge
)


MODEL_MAP = {
    'nano': (RFDETRNano, RFDETRSegNano),
    'small': (RFDETRSmall, RFDETRSegSmall),
    'medium': (RFDETRMedium, RFDETRSegMedium),
    'large': (RFDETRLarge, RFDETRSegLarge),
    'xlarge': (None, RFDETRSegXLarge),
    '2xlarge': (None, RFDETRSeg2XLarge),
}


def parse_args():
    parser = argparse.ArgumentParser(description='RF-DETR training script')
    parser.add_argument('--data', type=str, required=True, help='path to dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='number of training epochs')
    parser.add_argument('--batch', type=int, default=4, help='batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='image size')
    parser.add_argument('--device', type=str, default='0', help='device')
    parser.add_argument('--project', type=str, default='ckpts', help='project name')
    parser.add_argument('--name', type=str, default='train', help='run name')
    parser.add_argument('--resume', type=str, default=None, help='resume from checkpoint')
    parser.add_argument('--workers', type=int, default=4, help='number of workers')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--grad_accum_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--model', type=str, default='medium',
                       choices=['nano', 'small', 'medium', 'large', 'xlarge', '2xlarge'],
                       help='model size')
    parser.add_argument('--pretrained', type=str, default=None, help='path to pretrained checkpoint')
    return parser.parse_args()


def train(args):
    task = args.name.split('/')[0] if '/' in args.name else 'detect'
    is_segment = 'seg' in task.lower()

    model_pair = MODEL_MAP.get(args.model, (RFDETRMedium, RFDETRSegMedium))
    if is_segment:
        model_class = model_pair[1] if model_pair[1] else model_pair[0]
    else:
        model_class = model_pair[0] if model_pair[0] else model_pair[1]

    if model_class is None:
        raise ValueError(f"Model size '{args.model}' is not supported for task '{task}'")

    model_kwargs = {}
    if args.pretrained:
        model_kwargs['pretrain_weights'] = args.pretrained
    if args.resume:
        model_kwargs['resume'] = args.resume

    model = model_class(**model_kwargs)

    train_kwargs = {
        'dataset_dir': args.data,
        'epochs': args.epochs,
        'batch_size': args.batch,
        'output_dir': f"{args.project}/{args.name}",
        'lr': args.lr,
        'grad_accum_steps': args.grad_accum_steps,
        'num_workers': args.workers,
    }

    model.train(**train_kwargs)


if __name__ == "__main__":
    train(parse_args())