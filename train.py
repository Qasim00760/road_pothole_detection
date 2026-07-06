import torch
from ultralytics import YOLO
import shutil
import os
import csv
import sys

# Reconfigure stdout to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("==================================================")
    print("⚙️ YOLOv8 Pothole Model Fine-Tuning Script")
    print("==================================================")

    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training on: {device}")
    
    # Check if GPU is unavailable and warn user
    if device == 'cpu':
        print("[!] Warning: CUDA is not available. Training on CPU will be slow.")

    # Load base model (yolov8n.pt).
    # If the user already has it in the workspace root 'Models/yolov8n.pt', copy it first to save download time.
    base_model_source = os.path.join("..", "Models", "yolov8n.pt")
    if os.path.exists(base_model_source) and not os.path.exists("yolov8n.pt"):
        print(f"[*] Copying yolov8n.pt from workspace cache '{base_model_source}'...")
        try:
            shutil.copy2(base_model_source, "yolov8n.pt")
        except Exception as e:
            print(f"[!] Warning: Could not copy yolov8n.pt: {e}")

    try:
        model = YOLO('yolov8n.pt')
    except Exception as e:
        print(f"[-] Error loading base model yolov8n.pt: {e}")
        sys.exit(1)

    print("[*] Starting training...")
    try:
        # Train with exact settings specified
        results = model.train(
            data='data.yaml',
            epochs=50,
            imgsz=640,
            batch=16,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005,
            patience=15,
            save=True,
            save_period=10,
            project='runs',
            name='pothole_detector',
            exist_ok=True,
            device=device,
            verbose=True,
            # High-recall data augmentations to detect small, distant, and partial potholes
            hsv_v=0.4,       # brightness variations (sun/shadows)
            hsv_s=0.4,       # saturation variations
            scale=0.5,       # random scaling (simulates distant and close-up views)
            mosaic=1.0,      # combine images to train on smaller object sizes
            degrees=15.0,    # rotation (simulates skewed camera angles)
            translate=0.1,   # translation
            fliplr=0.5       # horizontal flips
        )
        
        print("[+] Training completed successfully.")
        
    except Exception as e:
        print(f"[-] Training failed with error: {e}")
        print("Please check that prepare_dataset.py was run and dataset/ directory is fully structured.")
        sys.exit(1)

    # Auto-copy best weights to models/ folder
    # YOLOv8 on Windows sometimes saves under runs/detect/runs/<name>/ instead of runs/<name>/
    # Check both paths and use whichever exists
    candidate_paths = [
        os.path.join('runs', 'pothole_detector', 'weights', 'best.pt'),
        os.path.join('runs', 'detect', 'runs', 'pothole_detector', 'weights', 'best.pt'),
        os.path.join('runs', 'detect', 'pothole_detector', 'weights', 'best.pt'),
    ]
    best_weights_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            best_weights_path = p
            break

    target_weights_dir = 'models'
    target_weights_path = os.path.join(target_weights_dir, 'best.pt')
    
    if os.path.exists(best_weights_path):
        os.makedirs(target_weights_dir, exist_ok=True)
        try:
            shutil.copy(best_weights_path, target_weights_path)
            print(f"✅ Best model saved to {target_weights_path}")
        except Exception as e:
            print(f"[-] Error copying weights to {target_weights_path}: {e}")
    else:
        print(f"[-] Error: Trained weights not found at {best_weights_path}")
        sys.exit(1)

    # Print training summary by parsing results.csv
    csv_path = 'runs/pothole_detector/results.csv'
    if os.path.exists(csv_path):
        try:
            with open(csv_path, mode='r') as f:
                reader = csv.DictReader(f)
                # Strip whitespace from dictionary keys and values
                rows = []
                for r in reader:
                    row_stripped = {k.strip(): v.strip() for k, v in r.items() if k is not None}
                    rows.append(row_stripped)
                
                # Retrieve final epoch metrics
                if rows:
                    final_row = rows[-1]
                    print("\n================ TRAINING SUMMARY ================")
                    print(f"Final mAP50: {final_row.get('metrics/mAP50(B)', 'N/A')}")
                    print(f"Final mAP50-95: {final_row.get('metrics/mAP50-95(B)', 'N/A')}")
                    
                    # Find the best epoch based on metrics/mAP50(B)
                    best_epoch = -1
                    best_map50 = -1.0
                    best_map50_95 = 0.0
                    for row in rows:
                        try:
                            epoch_val = int(row.get('epoch', '0'))
                            map50_val = float(row.get('metrics/mAP50(B)', '0.0'))
                            map50_95_val = float(row.get('metrics/mAP50-95(B)', '0.0'))
                        except ValueError:
                            continue
                        
                        if map50_val > best_map50:
                            best_map50 = map50_val
                            best_map50_95 = map50_95_val
                            best_epoch = epoch_val
                            
                    print(f"Best Epoch: {best_epoch} (mAP50: {best_map50:.4f}, mAP50-95: {best_map50_95:.4f})")
                    print("==================================================")
                else:
                    print("[-] Error: results.csv is empty.")
        except Exception as csv_err:
            print(f"[!] Warning: Could not parse results.csv for summary: {csv_err}")
    else:
        print(f"[!] Warning: results.csv not found at {csv_path}")

if __name__ == "__main__":
    main()
