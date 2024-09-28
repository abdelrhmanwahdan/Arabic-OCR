import os
import json
import cv2
import random
from sklearn.model_selection import train_test_split

# Paths
ann_dir = './Book/ann'  # Path to the annotations folder
img_dir = './Book/img'  # Path to the images folder
output_dir = './yolo_data'  # Path to save YOLO dataset (images and labels)

# Create directories if not existing
train_image_dir = os.path.join(output_dir, 'images/train')
val_image_dir = os.path.join(output_dir, 'images/val')
train_label_dir = os.path.join(output_dir, 'labels/train')
val_label_dir = os.path.join(output_dir, 'labels/val')

os.makedirs(train_image_dir, exist_ok=True)
os.makedirs(val_image_dir, exist_ok=True)
os.makedirs(train_label_dir, exist_ok=True)
os.makedirs(val_label_dir, exist_ok=True)


def convert_to_yolo_format(size, box):
    """
    Converts bounding box coordinates to YOLO format.

    Parameters:
    size (tuple): The width and height of the image (width, height).
    box (tuple): The bounding box coordinates in the format (x_min, y_min, x_max, y_max).

    Returns:
    tuple: Normalized coordinates in the format (x_center, y_center, width, height).
    """
    
    dw = 1. / size[0]
    dh = 1. / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    width = box[2] - box[0]
    height = box[3] - box[1]
    x_center *= dw
    width *= dw
    y_center *= dh
    height *= dh
    return x_center, y_center, width, height


def get_bounding_box(points):
    """
    Extracts bounding box coordinates from a list of polygon points.

    Parameters:
    points (list): List of tuples representing the points of a polygon (x, y).

    Returns:
    tuple: Bounding box coordinates in the format (x_min, y_min, x_max, y_max).
    """
    
    x_coordinates = [point[0] for point in points]
    y_coordinates = [point[1] for point in points]
    x_min = min(x_coordinates)
    x_max = max(x_coordinates)
    y_min = min(y_coordinates)
    y_max = max(y_coordinates)
    return x_min, y_min, x_max, y_max

# Get the list of annotation files
ann_files = [f for f in os.listdir(ann_dir) if f.endswith('.json')]

# Split the dataset into training and validation (80% train, 20% val)
train_files, val_files = train_test_split(ann_files, test_size=0.2, random_state=42)


def process_files(files, image_output_dir, label_output_dir):
    """
    Processes annotation files, converts bounding boxes to YOLO format, and saves results.

    Parameters:
    files (list): List of annotation files (JSON format).
    image_output_dir (str): Directory to save the processed images.
    label_output_dir (str): Directory to save the YOLO format label files.

    Returns:
    None
    """
    
    for ann_file in files:
        # Load the JSON file
        with open(os.path.join(ann_dir, ann_file), 'r') as f:
            ann_data = json.load(f)
        
        # Get corresponding image file (replace .json with image extension)
        image_file = ann_file.replace('.json', '')
        img_path = None
        for ext in ['.jpg', '.jpeg', '.png']:
            if os.path.exists(os.path.join(img_dir, image_file + ext)):
                img_path = os.path.join(img_dir, image_file + ext)
                break
        
        if img_path is None:
            print(f"Image not found for {ann_file}")
            continue
        
        # Load image to get dimensions
        img = cv2.imread(img_path)
        img_height, img_width = img.shape[:2]
        
        # Prepare output label file
        label_file = os.path.join(label_output_dir, ann_file.replace('.json', '.txt'))
        with open(label_file, 'w') as label_f:
            # Iterate over all objects
            for obj in ann_data['objects']:
                class_name = obj['classTitle']
                # We're interested in the 'Title' class only
                if class_name == 'Title':
                    points = obj['points']['exterior']
                    # Handle both polygons and rectangles
                    if len(points) == 2:  # It's a rectangle
                        x_min, y_min = points[0]
                        x_max, y_max = points[1]
                    else:  # It's a polygon
                        x_min, y_min, x_max, y_max = get_bounding_box(points)
                    
                    bbox = (x_min, y_min, x_max, y_max)
                    yolo_bbox = convert_to_yolo_format((img_width, img_height), bbox)
                    # YOLO format: class_index x_center y_center width height
                    label_f.write(f"0 {yolo_bbox[0]} {yolo_bbox[1]} {yolo_bbox[2]} {yolo_bbox[3]}\n")  # Class 0 for 'Title'

        # Copy the image to the appropriate train/val directory
        cv2.imwrite(os.path.join(image_output_dir, os.path.basename(img_path)), img)


# Process training files and save them to train directories
process_files(train_files, train_image_dir, train_label_dir)

# Process validation files and save them to validation directories
process_files(val_files, val_image_dir, val_label_dir)

print("Dataset prepared for YOLO training, with train/val split.")
