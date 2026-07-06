import os
import shutil
import cv2
import numpy as np
from ultralytics import YOLO

def load_model(model_path='models/best.pt'):
    """
    Load YOLO model from path.
    Print class names.
    Return model object.
    Raise FileNotFound error if file not found.
    """
    # Auto-copy model from workspace root Models cache if best.pt is not present in local models/
    if not os.path.exists(model_path):
        parent_model = os.path.join("..", "Models", "best.pt")
        if os.path.exists(parent_best := os.path.abspath(parent_model)):
            print(f"[*] Found pre-trained best.pt at '{parent_best}'. Copying to local path '{model_path}'...")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            try:
                shutil.copy2(parent_best, model_path)
            except Exception as e:
                print(f"[!] Warning: Could not copy pre-trained model: {e}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"[-] Error: YOLO model weight file not found at '{model_path}'.")

    try:
        model = YOLO(model_path)
        print("[+] YOLO model loaded successfully.")
        
        # Print class names
        class_names = model.names
        print(f"[*] Model Classes: {class_names}")
        return model
    except Exception as e:
        raise RuntimeError(f"[-] Failed to load YOLO model: {e}")

def enhance_road_contrast(image_np):
    """
    Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to the L channel
    of the LAB color space to enhance road contrast and texture, keeping colors intact.
    """
    try:
        lab = cv2.cvtColor(image_np, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return enhanced
    except Exception as e:
        print(f"[!] Warning: Contrast enhancement failed, returning original image: {e}")
        return image_np

def run_detection(model, image_np, conf_threshold=0.25, iou_threshold=0.45, imgsz=800, enhance_contrast=False):
    """
    Run YOLO inference on numpy image.
    Apply confidence, IOU, resolution thresholds, and contrast enhancement.
    Return list of dicts.
    """
    try:
        # Preprocess image with CLAHE if contrast enhancement is requested
        proc_img = enhance_road_contrast(image_np) if enhance_contrast else image_np
        
        # Run inference using model predict
        results = model.predict(
            source=proc_img,
            conf=conf_threshold,
            iou=iou_threshold,
            imgsz=imgsz,
            device='cpu', # default to CPU inference for streamlit stability
            verbose=False
        )
        
        detections = []
        if len(results) > 0:
            boxes = results[0].boxes
            for box in boxes:
                # Get coordinates
                xyxy = box.xyxy[0].cpu().numpy().tolist()
                conf = float(box.conf[0].cpu().numpy())
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = model.names.get(cls_id, 'pothole') # default to 'pothole'
                
                detections.append({
                    'bbox': xyxy,
                    'confidence': conf,
                    'class_name': class_name,
                    'class_id': cls_id
                })
        return detections
    except Exception as e:
        print(f"[-] Error running model inference: {e}")
        return []

def draw_detections(image_np, detections):
    """
    Draw bounding boxes on image:
    1. Filled semi-transparent red rectangle inside bbox (alpha=0.2)
    2. Solid red border (thickness=3): color (0, 0, 255)
    3. Label above box: "Pothole: 87.3%" in white text
    4. Black filled background behind label text
    Returns annotated image as numpy BGR array.
    """
    # Create copy of image to prevent drawing on the original
    annotated = image_np.copy()
    
    # Step 1: Draw filled red rectangles for alpha blending
    overlay = annotated.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), -1)
        
    # Blend overlay with original using alpha = 0.2
    cv2.addWeighted(overlay, 0.2, annotated, 0.8, 0, annotated)
    
    # Step 2: Draw solid red borders and label overlays
    for det in detections:
        x1, y1, x2, y2 = map(int, det['bbox'])
        
        # Solid border (BGR: 0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        
        # Text label configuration
        conf_percent = det['confidence'] * 100
        label_text = f"Pothole: {conf_percent:.1f}%"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        font_thickness = 1
        
        # Get dimensions of label text
        (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, font_thickness)
        
        # Handle position: placing above box, check if enough room, else place inside top-left
        if y1 - text_h - 10 > 0:
            label_y = y1 - 6
        else:
            label_y = y1 + text_h + 6
            
        label_x = x1
        
        # Background coords
        bg_x1 = label_x
        bg_y1 = label_y - text_h - 4
        bg_x2 = label_x + text_w + 4
        bg_y2 = label_y + baseline - 2
        
        # Draw black background card for text
        cv2.rectangle(annotated, (bg_x1, bg_y1), (bg_x2, bg_y2), (0, 0, 0), -1)
        
        # Draw white text (BGR: 255, 255, 255)
        cv2.putText(annotated, label_text, (label_x + 2, label_y - 2), font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
        
    return annotated

def get_detection_summary(detections, image_shape):
    """
    Return summary statistics:
    total_potholes: count
    avg_confidence: average confidence (0.0 if 0 potholes)
    max_confidence: max confidence (0.0 if 0 potholes)
    min_confidence: min confidence (0.0 if 0 potholes)
    severity: Low/Medium/High/None
    coverage_percent: area of bounding boxes divided by image area * 100
    """
    total = len(detections)
    
    if total == 0:
        return {
            'total_potholes': 0,
            'avg_confidence': 0.0,
            'max_confidence': 0.0,
            'min_confidence': 0.0,
            'severity': 'None',
            'coverage_percent': 0.0
        }
        
    confidences = [det['confidence'] for det in detections]
    avg_conf = float(np.mean(confidences)) * 100.0
    max_conf = float(np.max(confidences)) * 100.0
    min_conf = float(np.min(confidences)) * 100.0
    
    # Severity assessment
    if total == 0:
        severity = "None"
    elif 1 <= total <= 2:
        severity = "Low"
    elif 3 <= total <= 5:
        severity = "Medium"
    else:
        severity = "High"
        
    # Image resolution info
    height, width = image_shape[0], image_shape[1]
    img_area = height * width
    
    # Sum bbox areas
    bbox_area_sum = 0
    for det in detections:
        x1, y1, x2, y2 = det['bbox']
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        bbox_area_sum += (w * h)
        
    coverage = (bbox_area_sum / img_area) * 100.0
    
    return {
        'total_potholes': total,
        'avg_confidence': avg_conf,
        'max_confidence': max_conf,
        'min_confidence': min_conf,
        'severity': severity,
        'coverage_percent': coverage
    }
