import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import easyocr


class PlateDetector:
    def __init__(self, model_path: str = "best.pt"):
        """Initialize YOLO model and EasyOCR reader."""
        self.model = YOLO(model_path)
        # Initialize EasyOCR reader for Arabic and English/Numbers
        self.reader = easyocr.Reader(['ar', 'en'], gpu=False)

    def detect_and_ocr(self, image: Image.Image, conf_threshold: float = 0.25):
        """
        Detect license plates in the input PIL image, draw bounding boxes,
        and perform OCR on the cropped plate regions.
        """
        # Convert PIL Image to OpenCV BGR image
        img_np = np.array(image)
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        annotated_img = img_bgr.copy()

        results = self.model.predict(source=img_bgr, conf=conf_threshold)
        detections = []

        if len(results) > 0 and len(results[0].boxes) > 0:
            boxes = results[0].boxes

            for idx, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                # Crop plate area for OCR
                h, w, _ = img_bgr.shape
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                
                crop_x1 = max(0, x1 - pad_x)
                crop_y1 = max(0, y1 - pad_y)
                crop_x2 = min(w, x2 + pad_x)
                crop_y2 = min(h, y2 + pad_y)

                cropped_plate = img_bgr[crop_y1:crop_y2, crop_x1:crop_x2]

                # Perform OCR on cropped region
                ocr_results = self.reader.readtext(cropped_plate, detail=0)
                extracted_text = " ".join(ocr_results) if ocr_results else "No text detected"

                # Draw bounding box and label on image
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"Plate #{idx + 1} ({conf:.2f})"
                cv2.putText(
                    annotated_img,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Convert cropped plate to RGB for Streamlit display
                cropped_rgb = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

                detections.append({
                    "id": idx + 1,
                    "bbox": (x1, y1, x2, y2),
                    "confidence": conf,
                    "crop": cropped_rgb,
                    "text": extracted_text,
                })

        # Convert back to RGB for displaying in Streamlit
        annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        return annotated_rgb, detections