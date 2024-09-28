# Arabic Book Title Detection and OCR API

This project provides a FastAPI-based web service for detecting Arabic book titles in images and extracting the text using Optical Character Recognition (OCR). The service is built using a YOLOv8 model fine-tuned for detecting book titles and an Arabic OCR pipeline using EasyOCR, with spell and language correction performed using SpellChecker and a BERT-based model.

## Features

- **Book Title Detection**: The API detects Arabic book titles in images using a fine-tuned YOLOv8 model trained specifically to identify 'Title' as a class from a dataset.
- **OCR (Optical Character Recognition)**: After detecting the 'Title' regions, the API extracts the text from these detected regions using EasyOCR, which supports Arabic and recognizes text within the region of interest (ROI).

- **Spelling Correction**: Extracted text is passed through a spell-checking algorithm using the `SpellChecker` library to ensure the text is corrected for common spelling errors.

- **BERT-based Language Model Correction**: Further correction of the recognized text is done using a BERT-based masked language model (`aubmindlab/bert-base-arabertv02`). This model predicts masked tokens and corrects language-specific errors, improving the quality of OCR results.

- **Multi-Precision YOLO Model Support**: The YOLO model has been exported in three formats (PyTorch, ONNX, FP16) to compare performance across different inference settings. The API uses the model to identify book titles and returns the most confident results.

- **Image Saving**: The extracted regions of interest (ROI) from the image, containing the detected titles, are saved locally in the `predictions/` directory for further reference. This ensures all processed ROIs are stored and not overwritten.

- **REST API for Object Detection and OCR**: The FastAPI-based REST API provides an endpoint `/predict/` where users can upload an image, and the API will detect the title, extract and correct the text, and return a response with the bounding box, corrected text, and confidence scores.

- **Comparison of Different Model Precision Formats**: The API enables comparison of object detection results from models in different precision formats (PyTorch, ONNX, and FP16), providing performance metrics like `Precision`, `Recall`, and `mAP@0.50:0.95` to evaluate the best format.

## Project Structure

- **app.py**:

  - The main entry point of the FastAPI application. It handles the `/predict/` endpoint which takes an image as input, performs object detection using YOLOv8, applies OCR for text recognition, and returns the processed results with confidence scores and corrected text.

- **ocr_manager.py**:

  - Manages the OCR (Optical Character Recognition) functionality using EasyOCR. This module handles the text extraction from the detected regions of interest (ROIs) and performs text correction using a spell checker and BERT-based language model.

- **yolo_manager.py**:

  - Handles object detection using a pre-trained or fine-tuned YOLOv8 model. It loads the model, processes the input images, and identifies objects in the image, specifically focusing on detecting the "Title" class in the dataset.

- **Dockerfile**:

  - Defines the steps to build the Docker image for the application. It sets up the environment by installing the necessary system dependencies, copying project files, installing Python dependencies from `requirements.txt`, and exposing port `8000` for the FastAPI service.

- **requirements.txt**:

  - A list of Python dependencies that are required to run the project. This file can be used to install all the required libraries using `pip install -r requirements.txt`.

- **dataset.yaml**:

  - The dataset configuration file used by YOLOv8. It specifies the number of classes (`nc`) and the names of those classes, in this case, `'Title'`. It also contains the paths to the training and validation images, which are necessary for training or fine-tuning the YOLOv8 model.

- **predictions/**:

  - A directory that stores the region of interest (ROI) images that are extracted during object detection. These ROI images are processed by the OCR for text recognition. This folder is created dynamically if it doesn't exist.

- **model_cache/**:

  - Stores cached models, such as the YOLOv8 `.pt` model used for object detection. This directory ensures that the model does not need to be reloaded repeatedly.

- **quantized_models/**:

  - Contains the quantized versions of the YOLO models (e.g., ONNX or FP16). These models are optimized for faster inference while trading off some accuracy.

- **yolo_data/**:

  - Contains the images and labels used for YOLO model training and evaluation. This is referenced by `dataset.yaml` to specify the training and validation image paths.

- **runs/**:

  - A directory where YOLO saves its inference and training results, including logs, images, and evaluation metrics.

- **prepare_yolo_data.py**:
  - A utility script used to process and prepare the dataset in a format suitable for YOLOv8 training. It updates annotations and organizes the dataset paths.

## Requirements

- **Python 3.10 or later**:

  - The project is built using Python, and it requires version 3.10 or later to run successfully. Make sure you have the correct version installed. You can check your Python version by running:
    ```bash
    python --version
    ```

- **FastAPI**:

  - The web framework used for building the REST API is FastAPI, which offers high performance for asynchronous tasks and is easy to use. Install FastAPI using:
    ```bash
    pip install fastapi
    ```

- **Uvicorn**:

  - Uvicorn is an ASGI server used to run FastAPI apps. Install it using:
    ```bash
    pip install uvicorn
    ```

- **Torch**:

  - PyTorch is required for loading and running the YOLO model and BERT-based corrections. Install the required version for your setup:
    ```bash
    pip install torch
    ```

- **Ultralytics YOLOv8**:

  - The Ultralytics YOLO package is needed to run the object detection model (YOLOv8). Install using:
    ```bash
    pip install ultralytics
    ```

- **EasyOCR**:

  - EasyOCR is used to perform optical character recognition (OCR) on the detected book titles in the image. Install it using:
    ```bash
    pip install easyocr
    ```

- **Hugging Face Transformers**:

  - The `transformers` library from Hugging Face is used for BERT-based language model corrections. Install it using:
    ```bash
    pip install transformers
    ```

- **SpellChecker**:

  - The `pyspellchecker` package is used to correct spelling in extracted OCR text. Install it using:
    ```bash
    pip install pyspellchecker
    ```

- **OpenCV**:

  - OpenCV is used for image manipulation, such as extracting regions of interest (ROI) from the images for OCR. Install it using:
    ```bash
    pip install opencv-python
    ```

- **Numpy**:

  - Numpy is used for handling image data as arrays for processing. Install it using:
    ```bash
    pip install numpy
    ```

- **Docker (Optional)**:

  - If you prefer to run the project in a Docker container, make sure you have Docker installed. You can find the installation guide [here](https://docs.docker.com/get-docker/).
  - After installing Docker, build the Docker image for the project by running:
    ```bash
    docker build -t ocr-app .
    ```

- **Additional System Dependencies** (Installed in Docker):

  - **FFmpeg**: Required for video or multimedia handling (if needed). Installed in the Dockerfile as part of the image setup:

    ```bash
    apt-get update && apt-get install ffmpeg libsm6 libxext6 -y
    ```

  - **libgirepository, libcairo, etc.**: Required for GTK-based functionalities like image processing. These are system dependencies installed within the Docker container to support specific functionalities like image recognition.

---

### Installation Command

To install all the Python dependencies in one go, you can use the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

## Model Comparison: Performance of Different Models

The YOLOv8 model has been tested in three versions: the original PyTorch model, an ONNX version, and an FP16 version. Below is a comparison of their performance:

| Model Name         | Format  | mAP@50-95 | Precision | Recall |
| ------------------ | ------- | --------- | --------- | ------ |
| Fine-tuned YOLOv8m | PyTorch | 0.5687    | 0.6803    | 0.7192 |
| Fine-tuned YOLOv8m | ONNX    | 0.5741    | 0.6950    | 0.6901 |
| Fine-tuned YOLOv8m | FP16    | 0.5687    | 0.6803    | 0.7192 |

## Setting Up the Project

### Clone the Repository:

First, clone the project repository and navigate into it:

```bash
git clone https://github.com/yourusername/Arabic-Ocr-Detection-Api.git
cd Arabic-Ocr-Detection-Api
```

### Running the Application with Docker:

You can containerize the project and run it using Docker, which ensures consistency across environments without manually installing dependencies.

1. **Build the Docker Image**:

   Build the Docker image using the `Dockerfile` included in the project:

   ```bash
   docker build -t ocr-app .
   ```

2. **Run the Docker Container**:

   Once the Docker image is built, you can run the container with the following command:

   ```bash
   docker run -d -p 8000:8000 --name ocr-app-container ocr-app
   ```

   After running the container, the API will be accessible at `http://localhost:8000`.

## API Usage:

The main functionality of the project is exposed through the `/predict/` endpoint. This API endpoint allows you to upload images of book titles and returns the detected and corrected text.

### Example API Request:

Here’s how you can interact with the API using `curl`. Replace `/path/to/your/image.jpg` with the path to your actual image file:

```bash
curl -X POST "http://localhost:8000/predict/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.jpg"
```

### Example API Response:

The API will return a JSON object containing the results of the object detection and OCR:

```json
{
  "bounding_box": [489, 311, 1094, 444],
  "detected_text": "شياء تؤل",
  "corrected_text_spell": "شياء تقل",
  "corrected_text_bert": "شيا لم تقل",
  "confidence": 0.7135,
  "roi_image_path": "./predictions/roi_20240927_181106_489_311.jpg"
}
```

#### Response Details:

- **bounding_box**: The coordinates of the detected book title in the image.
- **detected_text**: The original text detected by EasyOCR.
- **corrected_text_spell**: The spell-corrected version of the detected text.
- **corrected_text_bert**: The BERT-corrected version of the text.
- **confidence**: The confidence score from the object detection model.
- **roi_image_path**: Path to the image containing the detected region of interest (ROI), saved during the inference process.

With this setup, you can now detect and correct Arabic text in book titles and run the app either locally or in a Docker container.

## Training and Evaluating the Model

### 1. Training:

To train the YOLOv8 model using a custom dataset:

```bash
python yolo_manager.py train --data-path dataset.yaml --epochs 100
```

### 2. Evaluation:

To evaluate the model:

```bash
python yolo_manager.py evaluate --model-path ./models/best.pt --data-path dataset.yaml
```

### 3. Exporting:

To export the YOLO model to ONNX or FP16 formats:

```bash
python yolo_manager.py convert --model-path ./models/best.pt --precision onnx --save-dir ./quantized_models
python yolo_manager.py convert --model-path ./models/best.pt --precision fp16 --save-dir ./quantized_models
```

## Directory Structure

```
.
├── app.py                 # FastAPI app file
├── ocr_manager.py         # Manages OCR and text correction logic
├── yolo_manager.py        # Manages YOLOv8 model for object detection
├── Dockerfile             # Docker setup file
├── requirements.txt       # Python dependencies
├── dataset.yaml           # YOLO dataset configuration
├── predictions/           # Folder for storing ROI images
└── models/                # Folder to store YOLO models
```

## Potential Improvements and Future Work

While the current implementation achieves solid performance, several enhancements could be made to improve both accuracy and efficiency:

- **Longer Training Period**:  
  The model was only trained for 10 epochs due to time and computational limitations. Increasing the number of training epochs would allow the model to better generalize from the data, improving accuracy and recall. This would be particularly beneficial in refining the detection of book titles, as the model would have more time to learn the intricate features in the data.

- **VisionEncoderDecoder with AraT5**:  
  To further improve OCR accuracy, we could explore using VisionEncoderDecoder models, which combine a Vision Transformer (ViT) as the encoder and a text-generating model like AraT5 as the decoder. This architecture could offer more sophisticated text generation capabilities, particularly for Arabic text. Here's a snippet of how the architecture could be adjusted:

  ```python
  from transformers import VisionEncoderDecoderModel, TrOCRProcessor, AutoTokenizer, ViTImageProcessor

  # Load the AraT5 tokenizer for Arabic text
  tokenizer = AutoTokenizer.from_pretrained("akhooli/araT5-base")

  # Load the ViTImageProcessor (formerly ViTFeatureExtractor) for ViT image preprocessing
  image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

  # Combine the feature extractor and tokenizer into a processor
  processor = TrOCRProcessor(feature_extractor=image_processor, tokenizer=tokenizer)

  # Load the VisionEncoderDecoder model with a ViT encoder and AraT5 decoder
  model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained("google/vit-base-patch16-224", "akhooli/araT5-base")
  ```

  This change would leverage the power of transformers for both image processing and natural language generation, allowing for potentially higher accuracy in text extraction from complex images.

- **Improved Data Augmentation**:  
  By incorporating more advanced data augmentation techniques, such as rotations, brightness variations, and noise, we can further improve the robustness of the YOLO model. This would help the model generalize better to different types of book titles in varying conditions.

- **Exploring Quantization Strategies**:  
  We applied dynamic quantization to reduce the model size and inference time. However, experimenting with other quantization strategies like post-training quantization or mixed-precision training could yield further performance improvements, especially in low-resource environments.

## Conclusion

This API allows you to detect Arabic book titles from images and extract the corrected text using a combination of YOLOv8 for object detection and EasyOCR with additional spell checking and language correction. Additionally, the performance of the model in PyTorch, ONNX, and FP16 formats has been compared, with ONNX providing slightly better precision.