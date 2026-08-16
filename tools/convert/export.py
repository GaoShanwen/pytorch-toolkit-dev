import argparse
import os
from pathlib import Path

from rfdetr import RFDETR


def parse_args():
    parser = argparse.ArgumentParser(description="Export RF-DETR model to ONNX format")
    parser.add_argument("--weight-path", type=str, required=True, help="path to checkpoint")
    parser.add_argument("--output-dir", type=str, default=None, help="output directory for exported model")
    parser.add_argument("--output-name", type=str, default=None, help="output filename (without extension)")
    parser.add_argument("--imgsz", type=int, default=[384, 640], help="input image size")
    parser.add_argument("--batch-size", type=int, default=1, help="batch size for export")
    parser.add_argument("--dynamic-batch", action="store_true", default=False, help="export with dynamic batch dimension")
    parser.add_argument("--fp16", action="store_true", default=False, help="export with FP16 precision")
    parser.add_argument("--trust-checkpoint", action="store_true", default=False, help="allow unsafe checkpoint deserialization")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if not os.path.exists(args.weight_path):
        raise FileNotFoundError(f"Weight file not found: {args.weight_path}")

    if args.output_dir is None:
        output_dir = Path(args.weight_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    output_name = args.output_name
    if output_name is None:
        output_name = Path(args.weight_path).stem

    print(f"Loading model from {args.weight_path}...")
    model = RFDETR.from_checkpoint(args.weight_path, trust_checkpoint=args.trust_checkpoint)

    if isinstance(args.imgsz, int):
        args.imgsz = [args.imgsz, args.imgsz]
    print(f"Exporting to ONNX with imgsz={args.imgsz}, batch_size={args.batch_size}...")

    model.export(
        output_dir=str(output_dir),
        format="onnx",
        shape=args.imgsz,
        batch_size=args.batch_size,
        dynamic_batch=args.dynamic_batch,
        fp16=args.fp16,
    )

    exported_files = list(output_dir.glob("*.onnx"))
    if len(exported_files) == 1:
        exported_file = exported_files[0]
        if output_name and exported_file.name != f"{output_name}.onnx":
            target_path = output_dir / f"{output_name}.onnx"
            exported_file.rename(target_path)
            print(f"Exported model saved to: {target_path}")
        else:
            print(f"Exported model saved to: {exported_file}")
    elif len(exported_files) > 1:
        print(f"Multiple ONNX files exported: {exported_files}")