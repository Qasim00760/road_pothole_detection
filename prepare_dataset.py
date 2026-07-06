import os
import xml.etree.ElementTree as ET
import random
import shutil
import sys

# Reconfigure stdout to UTF-8 to prevent encoding errors on Windows terminal
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Raw dataset configuration (as requested)
RAW_IMAGES_DIR = "raw_dataset/images"
RAW_ANNOTATIONS_DIR = "raw_dataset/annotations"

# Target dataset structure
DATASET_DIR = "dataset"
IMAGES_TRAIN_DIR = os.path.join(DATASET_DIR, "images", "train")
IMAGES_VAL_DIR = os.path.join(DATASET_DIR, "images", "val")
LABELS_TRAIN_DIR = os.path.join(DATASET_DIR, "labels", "train")
LABELS_VAL_DIR = os.path.join(DATASET_DIR, "labels", "val")

def setup_environment():
    """Ensure raw_dataset is linked/available and create output directories."""
    # Check if raw_dataset exists. If not, auto-link it from parent directory to run out of the box
    if not os.path.exists("raw_dataset"):
        parent_dataset = os.path.join("..", "Road Pothole Detection System Dataset")
        if os.path.exists(parent_dataset):
            print(f"[*] Found raw dataset at '{parent_dataset}'. Creating symbolic link/junction 'raw_dataset'...")
            try:
                # Try creating a junction on Windows (no privileges needed)
                import subprocess
                subprocess.run(f'mklink /J "raw_dataset" "{parent_dataset}"', shell=True, check=True)
                print("[+] Successfully created raw_dataset directory junction.")
            except Exception as e:
                print(f"[!] Could not create junction link: {e}. Attempting directory copy...")
                try:
                    shutil.copytree(parent_dataset, "raw_dataset")
                    print("[+] Successfully copied dataset to 'raw_dataset'.")
                except Exception as copy_err:
                    print(f"[-] Failed to copy dataset: {copy_err}")
        else:
            print("[-] Warning: 'raw_dataset' folder not found. Please place dataset in raw_dataset/ or run from project root.")

    # Create target directories
    for d in [IMAGES_TRAIN_DIR, IMAGES_VAL_DIR, LABELS_TRAIN_DIR, LABELS_VAL_DIR]:
        os.makedirs(d, exist_ok=True)

def parse_xml_to_yolo(xml_path):
    """
    Parses a single VOC XML file, returns image filename, dimension, and bounding boxes.
    Bounding box format: (class_id, x_center, y_center, width, height)
    """
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Get filename
        filename_elem = root.find("filename")
        if filename_elem is not None:
            img_filename = filename_elem.text
        else:
            # Fallback to XML base name + .png
            img_filename = os.path.basename(xml_path).replace(".xml", ".png")
            
        # Get dimensions
        size_elem = root.find("size")
        if size_elem is None:
            print(f"[!] Warning: <size> tag missing in {xml_path}")
            return None, None, []
            
        width_elem = size_elem.find("width")
        height_elem = size_elem.find("height")
        if width_elem is None or height_elem is None:
            print(f"[!] Warning: width/height tag missing in {xml_path}")
            return None, None, []
            
        img_width = int(width_elem.text)
        img_height = int(height_elem.text)
        
        if img_width <= 0 or img_height <= 0:
            print(f"[!] Warning: Invalid image size ({img_width}x{img_height}) in {xml_path}")
            return None, None, []
            
        boxes = []
        # Find all objects with name = pothole
        for obj in root.findall("object"):
            name_elem = obj.find("name")
            if name_elem is not None and name_elem.text == "pothole":
                bndbox = obj.find("bndbox")
                if bndbox is not None:
                    xmin = float(bndbox.find("xmin").text)
                    ymin = float(bndbox.find("ymin").text)
                    xmax = float(bndbox.find("xmax").text)
                    ymax = float(bndbox.find("ymax").text)
                    
                    # Convert to YOLO format (0-1 normalized)
                    x_center = (xmin + xmax) / 2 / img_width
                    y_center = (ymin + ymax) / 2 / img_height
                    width    = (xmax - xmin) / img_width
                    height   = (ymax - ymin) / img_height
                    
                    # Clip coordinates to [0.0, 1.0] to prevent YOLO errors
                    x_center = max(0.0, min(1.0, x_center))
                    y_center = max(0.0, min(1.0, y_center))
                    width = max(0.0, min(1.0, width))
                    height = max(0.0, min(1.0, height))
                    
                    # class index for pothole is 0
                    boxes.append((0, x_center, y_center, width, height))
                    
        return img_filename, (img_width, img_height), boxes
        
    except Exception as e:
        print(f"[!] Error parsing XML file {xml_path}: {e}")
        return None, None, []

def main():
    print("==================================================")
    print("🚀 Pothole Dataset Preprocessor: VOC XML to YOLOv8")
    print("==================================================")
    
    setup_environment()
    
    if not os.path.exists(RAW_ANNOTATIONS_DIR) or not os.path.exists(RAW_IMAGES_DIR):
        print(f"[-] Error: Raw directories '{RAW_ANNOTATIONS_DIR}' or '{RAW_IMAGES_DIR}' do not exist.")
        sys.exit(1)
        
    # Get all XML files
    xml_files = [f for f in os.listdir(RAW_ANNOTATIONS_DIR) if f.endswith(".xml")]
    if not xml_files:
        print("[-] Error: No XML files found in raw annotations directory.")
        sys.exit(1)
        
    print(f"[*] Found {len(xml_files)} XML annotation files.")
    
    valid_records = []
    skipped_no_annotation = 0
    total_potholes = 0
    
    for xml_file in xml_files:
        xml_path = os.path.join(RAW_ANNOTATIONS_DIR, xml_file)
        img_filename, size, boxes = parse_xml_to_yolo(xml_path)
        
        if img_filename is None:
            continue
            
        # Check if corresponding image exists
        img_path = os.path.join(RAW_IMAGES_DIR, img_filename)
        if not os.path.exists(img_path):
            # Try case-insensitive matching or check with other extensions
            base_no_ext, _ = os.path.splitext(img_filename)
            matched = False
            for ext in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']:
                alt_path = os.path.join(RAW_IMAGES_DIR, base_no_ext + ext)
                if os.path.exists(alt_path):
                    img_path = alt_path
                    img_filename = base_no_ext + ext
                    matched = True
                    break
            
            if not matched:
                print(f"[!] Warning: Image file '{img_filename}' not found for '{xml_file}'. Skipping.")
                continue
                
        if not boxes:
            skipped_no_annotation += 1
            continue
            
        valid_records.append({
            'xml_file': xml_file,
            'img_filename': img_filename,
            'img_path': img_path,
            'boxes': boxes
        })
        total_potholes += len(boxes)

    print(f"[*] Total valid annotated files: {len(valid_records)}")
    print(f"[*] Total skipped (no potholes found): {skipped_no_annotation}")
    
    if not valid_records:
        print("[-] Error: No valid image-annotation pairs found to split.")
        sys.exit(1)
        
    # Shuffle and split 80% train, 20% val
    random.seed(42)
    random.shuffle(valid_records)
    
    split_idx = int(len(valid_records) * 0.8)
    train_records = valid_records[:split_idx]
    val_records = valid_records[split_idx:]
    
    # Process Train Set
    print("[*] Processing train set...")
    for rec in train_records:
        dest_img_path = os.path.join(IMAGES_TRAIN_DIR, rec['img_filename'])
        shutil.copy2(rec['img_path'], dest_img_path)
        
        # Save label text file
        label_filename = os.path.splitext(rec['img_filename'])[0] + ".txt"
        dest_label_path = os.path.join(LABELS_TRAIN_DIR, label_filename)
        with open(dest_label_path, 'w') as lf:
            for box in rec['boxes']:
                lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
    # Process Val Set
    print("[*] Processing validation set...")
    for rec in val_records:
        dest_img_path = os.path.join(IMAGES_VAL_DIR, rec['img_filename'])
        shutil.copy2(rec['img_path'], dest_img_path)
        
        # Save label text file
        label_filename = os.path.splitext(rec['img_filename'])[0] + ".txt"
        dest_label_path = os.path.join(LABELS_VAL_DIR, label_filename)
        with open(dest_label_path, 'w') as lf:
            for box in rec['boxes']:
                lf.write(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
                
    print("\n================ SUMMARY ================")
    print(f"Total images processed: {len(valid_records)}")
    print(f"Train set: {len(train_records)} images")
    print(f"Val set: {len(val_records)} images")
    print(f"Total potholes annotated: {total_potholes}")
    print(f"Skipped (no annotations): {skipped_no_annotation}")
    print("=========================================")
    print("[+] Dataset preparation completed successfully!")

if __name__ == "__main__":
    main()
