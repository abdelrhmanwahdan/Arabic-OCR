import os
import argparse
import shutil
import cv2
import json
from ultralytics import YOLO
import torch

class YOLOManager:
    """
    YOLOManager is a unified class to manage YOLO model training, inference, evaluation, caching,
    and other utilities such as annotations plotting and model conversion.

    Methods:
    - train_model: Trains a YOLO model, with caching functionality.
    - infer_images: Runs inference on a directory of images and saves output results.
    - evaluate_model: Evaluates a trained YOLO model and prints out precision, recall, and mAP metrics.
    - cache_model: Caches a model after training for later use.
    - load_cached_model: Loads a cached model if it exists.
    - convert_model: Converts a YOLO model to ONNX or TensorRT format with precision FP16.
    - plot_original_annotations: Plots the bounding boxes from original JSON annotations onto an image.
    - plot_yolo_annotations: Plots YOLO-format annotations on the image.
    """

    def __init__(self, model_path='yolov8m.pt', cache_dir='./model_cache'):
        """
        Initializes YOLOManager with a model path and cache directory.

        Args:
        - model_path (str): Path to the YOLO model. Default is 'yolov8m.pt'.
        - cache_dir (str): Directory to cache trained models for later use. Default is './model_cache'.
        """
        self.model_path = model_path
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.model = self.load_cached_model() or YOLO(model_path)

    def load_cached_model(self):
        """
        Loads a cached model if it exists in the cache directory.

        Returns:
        - model: Loaded YOLO model if it exists, otherwise None.
        """
        cached_model_path = os.path.join(self.cache_dir, 'best.pt')
        if os.path.exists(cached_model_path):
            print(f"Loading cached model from {cached_model_path}")
            return YOLO(cached_model_path)
        else:
            print("No cached model found.")
            return None

    def cache_model(self, model):
        """
        Caches the trained model for future use in the cache directory.

        Args:
        - model: YOLO model object to be cached.
        """
        cached_model_path = os.path.join(self.cache_dir, 'best.pt')
        model.save(cached_model_path)
        print(f"Model cached at {cached_model_path}")

    def train_model(self, data_path, imgsz=640, epochs=10, batch_size=2, name='yolov8m', workers=2):
        """
        Trains the YOLO model on a given dataset. If a cached model exists, it loads the model.

        Args:
        - data_path (str): Path to the dataset configuration file.
        - imgsz (int): Image size for training.
        - epochs (int): Number of training epochs.
        - batch_size (int): Batch size for training.
        - name (str): Name of the training run.
        - workers (int): Number of data loading workers.
        """
        if self.load_cached_model():
            print("Using cached model, skipping training.")
            return
        else:
            self.model.train(
                data=data_path,
                imgsz=imgsz,
                epochs=epochs,
                batch=batch_size,
                name=name,
                workers=workers
            )
            # Cache the best model after training
            self.cache_model(self.model)

    def infer_images(self, source_dir, imgsz=640):
        """
        Performs inference on a directory of images and saves the results.
        If a cached model exists, it uses the cached model for inference.

        Args:
        - source_dir (str): Directory containing the input images.
        - imgsz (int): Image size for inference.
        """
        model = self.load_cached_model() or self.model
        results = model.predict(
            source=source_dir,
            imgsz=imgsz,
            save=True
        )

    def evaluate_model(self, data_path, model_path=None, imgsz=640):
        """
        Evaluates a model on the validation dataset and prints evaluation metrics.
        If no model path is provided, the cached 'best.pt' model is used by default.

        Args:
        - data_path (str): Path to the dataset configuration file for evaluation.
        - model_path (str, optional): Path to the model file. Default is None, which uses the cached model (best.pt).
        - imgsz (int): Image size for evaluation. Default is 640.
        """
        # Use the provided model path or fallback to the cached 'best.pt' model
        model_path = model_path or './model_cache/best.pt'

        # Load the model based on the provided model path (PyTorch or ONNX)
        if model_path.endswith('.onnx'):
            print(f"Loading ONNX model: {model_path}")
            model = YOLO(model_path)
        else:
            print(f"Loading PyTorch model: {model_path}")
            model = YOLO(model_path)

        print(f"Evaluating model: {model_path}")

        # Perform the evaluation (validation) on the dataset
        results = model.val(
            data=data_path,
            imgsz=imgsz,
            name='evaluation'
        )

        # Extract evaluation metrics
        metrics = results.results_dict
        precision = metrics['metrics/precision(B)']
        recall = metrics['metrics/recall(B)']
        map50 = metrics['metrics/mAP50(B)']
        map95 = metrics['metrics/mAP50-95(B)']

        # Print the evaluation metrics
        print(f"Evaluation results for {model_path}:")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  mAP@0.50: {map50:.4f}")
        print(f"  mAP@0.50:0.95: {map95:.4f}")




    def convert_model(self, save_dir='./converted_models', precision='fp16'):
        """
        Converts and saves the YOLO model in fp16 and ONNX format.
        
        Args:
        - model_path (str): Path to the YOLO model file (e.g., best.pt).
        - save_dir (str): Directory where the converted models will be saved. Default is './converted_models'.
        - precision (str): Precision level for conversion. Default is 'fp16'. Supported values are 'fp16' and 'onnx'.
        """
        # Load the cached model or current model
        model = self.load_cached_model() or self.model

        # Create the save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using {device}")

        if precision == 'fp16':
            # Convert the model to fp16 precision
            print(f"Converting model to {precision} precision...")
            model.model.half()  # Change the model to half precision (fp16)

            # Save the fp16 model
            fp16_output_path = os.path.join(save_dir, 'yolov8m_fp16.pt')
            torch.save(model, fp16_output_path)
            print(f"Model (FP16) saved successfully at {fp16_output_path}!")

        elif precision == 'onnx':
            # Export the model to ONNX format
            print(f"Exporting model to ONNX format...")
            onnx_output = os.path.join('model_cache', 'best.onnx')  # Path where YOLO saves ONNX model
            self.model.export(format='onnx', device='cpu')  # Export the model

            # Move the ONNX file to the specified directory
            final_output_name = os.path.join(save_dir, f'yolov8m.onnx')
            shutil.move(onnx_output, final_output_name)
            print(f"ONNX model saved successfully at {final_output_name}!")

        else:
            raise ValueError(f"Unsupported precision: {precision}. Choose either 'fp16' or 'onnx'.")





    @staticmethod
    def get_bounding_box(points):
        """
        Extracts a bounding box from polygon points.

        Args:
        - points (list): List of [x, y] coordinates for the bounding box.

        Returns:
        - (tuple): Bounding box in (x_min, y_min, x_max, y_max) format.
        """
        x_coordinates = [point[0] for point in points]
        y_coordinates = [point[1] for point in points]
        return min(x_coordinates), min(y_coordinates), max(x_coordinates), max(y_coordinates)

    def plot_original_annotations(self, json_path, img_path):
        """
        Plots the bounding boxes from original JSON annotations onto an image.

        Args:
        - json_path (str): Path to the JSON annotation file.
        - img_path (str): Path to the original image file.
        """
        with open(json_path) as f:
            data = json.load(f)
        image = cv2.imread(img_path)

        for obj in data['objects']:
            class_name = obj['classTitle']
            points = obj['points']['exterior']
            x_min, y_min, x_max, y_max = self.get_bounding_box(points)
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, class_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite('output_image.jpg', image)

    def plot_yolo_annotations(self, image_path, annotations_path, class_names):
        """
        Plots YOLO-format annotations on the image.

        Args:
        - image_path (str): Path to the image file.
        - annotations_path (str): Path to the YOLO annotations file.
        - class_names (list): List of class names corresponding to class IDs.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {image_path}")
            return

        with open(annotations_path, 'r') as file:
            annotations = file.readlines()

        img_height, img_width, _ = image.shape
        for annotation in annotations:
            components = annotation.strip().split()
            class_id = int(components[0])
            x_center, y_center, width, height = map(float, components[1:])
            x_center *= img_width
            y_center *= img_height
            width *= img_width
            height *= img_height
            x_min = int(x_center - width / 2)
            y_min = int(y_center - height / 2)
            x_max = int(x_center + width / 2)
            y_max = int(y_center + height / 2)
            label = class_names[class_id]
            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imwrite('output_image2.jpg', image)

def parse_args():
    """
    Parses the arguments passed through the command line.
    """
    parser = argparse.ArgumentParser(description="Manage YOLO Operations (Training, Inference, Evaluation, etc.)")

    subparsers = parser.add_subparsers(dest='command', help='Sub-commands for different operations')

    # Subparser for 'train' command
    train_parser = subparsers.add_parser('train', help='Train YOLO model')
    train_parser.add_argument('--data-path', type=str, required=True, help='Path to dataset configuration file')
    train_parser.add_argument('--imgsz', type=int, default=640, help='Image size for training')
    train_parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
    train_parser.add_argument('--batch-size', type=int, default=2, help='Batch size for training')
    train_parser.add_argument('--name', type=str, default='yolov8m', help='Name for training run')
    train_parser.add_argument('--workers', type=int, default=2, help='Number of workers for data loading')

    # Subparser for 'infer' command
    infer_parser = subparsers.add_parser('infer', help='Infer images using YOLO model')
    infer_parser.add_argument('--source-dir', type=str, required=True, help='Directory of input images')
    infer_parser.add_argument('--imgsz', type=int, default=640, help='Image size for inference')

    # Subparser for 'evaluate' command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate YOLO model')
    eval_parser.add_argument('--data-path', type=str, required=True, help='Path to the dataset configuration file')
    eval_parser.add_argument('--model-path', type=str, default=None, help='Optional path to the model to evaluate. Defaults to best.pt')
    eval_parser.add_argument('--imgsz', type=int, default=640, help='Image size for evaluation')

    # Subparser for 'convert' command
    convert_parser = subparsers.add_parser('convert', help='Convert YOLO model')
    convert_parser.add_argument('--precision', choices=['fp16', 'onnx'], required=True, help='Conversion precision (fp16 or onnx)')
    convert_parser.add_argument('--save-dir', type=str, default='./converted_models', help='Directory to save converted models')
    
    # Model path argument (shared across subcommands)
    parser.add_argument('--model-path', type=str, default='yolov8m.pt', help='Path to the YOLO model')

    return parser.parse_args()

def main():
    args = parse_args()
    yolo_manager = YOLOManager(model_path=args.model_path)

    if args.command == 'train':
        yolo_manager.train_model(
            data_path=args.data_path,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch_size=args.batch_size,
            name=args.name,
            workers=args.workers
        )
    elif args.command == 'infer':
        yolo_manager.infer_images(
            source_dir=args.source_dir,
            imgsz=args.imgsz
        )
    elif args.command == 'evaluate':
        yolo_manager.evaluate_model(
            data_path=args.data_path,
            model_path=args.model_path,
            imgsz=args.imgsz
        )
    elif args.command == 'convert':
        yolo_manager.convert_model(
            precision=args.precision,
            save_dir=args.save_dir
        )

if __name__ == "__main__":
    main()
