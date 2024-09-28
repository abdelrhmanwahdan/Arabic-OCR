from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import numpy as np
import cv2
from ocr_manager import ArabicOCR
from yolo_manager import YOLOManager
import os
from datetime import datetime

# Initialize the FastAPI app
app = FastAPI()

# Initialize the OCR Manager
ocr = ArabicOCR()

# Initialize the YOLO Manager with the path to the YOLO model
yolo_manager = YOLOManager(model_path="./models/best.pt")

# Define the class index for 'Title'
TITLE_BOOK_CLASS_ID = 0  # Class 'Title' based on your dataset.yaml

# Define the directory to save predictions (ROI images)
PREDICTIONS_DIR = './predictions/'

# Ensure the predictions folder exists
if not os.path.exists(PREDICTIONS_DIR):
    os.makedirs(PREDICTIONS_DIR)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Perform YOLO object detection and OCR with spelling correction on the detected 'Title' class areas.

    Args:
    - file: Uploaded image file.

    Returns:
    - JSON response containing the corrected text with the highest confidence for 'Title' class.
    """
    
    # Read the uploaded image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform object detection using the YOLO model
    results = yolo_manager.model.predict(source=image)

    # Store the best OCR result with the highest confidence
    best_result = None

    # Loop over the detected objects from YOLO
    for det in results[0].boxes:
        x1, y1, x2, y2 = det.xyxy[0].tolist()  # Convert tensor to list for coordinates
        conf = det.conf.tolist()  # Confidence, this may still be a list
        cls = det.cls.tolist()  # Class ID

        # Handle case where cls is a list
        if isinstance(cls, list):
            cls = cls[0]  # Extract the first element if it's a list

        cls = int(cls)  # Convert class ID to an integer

        # Handle case where conf is a list
        if isinstance(conf, list):
            conf = conf[0]  # Extract the first element if it's a list

        conf = float(conf)  # Convert confidence to float

        # Check if the detected object is of class 'Title'
        if cls == TITLE_BOOK_CLASS_ID:
            roi = image[int(y1):int(y2), int(x1):int(x2)]  # Extract region of interest (ROI)

            # Save the ROI image in the predictions folder
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            roi_filename = f"roi_{timestamp}_{int(x1)}_{int(y1)}.jpg"
            roi_path = os.path.join(PREDICTIONS_DIR, roi_filename)
            cv2.imwrite(roi_path, roi)

            # Perform OCR on the saved ROI image
            ocr_results = ocr.perform_ocr(roi_path)
            
            # Loop through the OCR results and find the one with the highest confidence
            for (bbox, text, prob) in ocr_results:
                # Correct the detected text using a spell checker
                corrected_text_spell = ocr.correct_spelling(text)
                
                # Further correct the text using a BERT-based language model
                corrected_text_bert = ocr.correct_with_bert(corrected_text_spell)

                # If this is the first result or the confidence is higher than the current best
                if best_result is None or conf > best_result["confidence"]:
                    best_result = {
                        "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                        "detected_text": text,
                        "corrected_text_spell": corrected_text_spell,
                        "corrected_text_bert": corrected_text_bert,
                        "confidence": conf,  # Use the float confidence here
                        "roi_image_path": roi_path  # Include the path of the saved ROI image
                    }

    # If no 'Title' class detections were found
    if best_result is None:
        return JSONResponse(content={"message": "No 'Title' class detected."})

    # Return the best OCR and correction result as JSON response
    return JSONResponse(content=best_result)

if __name__ == "__main__":
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)
