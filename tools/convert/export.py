from ultralytics import YOLO
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight-path', type=str, required=True, help='weight path')
    parser.add_argument('--image-path', type=str, default="data/pose-dataset/Person/demo.jpeg", help='weight path')

    return parser.parse_args()


if __name__=="__main__":
    args = parse_args()
    # Load a pretrained YOLO26n model
    model = YOLO(args.weight_path, task="pose")    

    # # Evaluate the model's performance on the validation set
    # metrics = model.val()

    # # Perform object detection on an image
    # results = model(args.image_path)  # Predict on an image
    # results[0].show()  # Display results

    # Export the model to ONNX format for deployment
    path = model.export(format="onnx")  # Returns the path to the exported model
