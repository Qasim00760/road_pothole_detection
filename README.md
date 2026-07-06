# Road Pothole Detection System 🕳️

An AI-powered system designed to detect road potholes from images using a fine-tuned **YOLOv8** model and serve results through a professional, high-performance **Streamlit** dashboard.

## 🌟 Real-World Use Case

Road maintenance and safety are critical concerns for urban planning. Potholes damage vehicles, cause accidents, and lead to high maintenance costs. This system can be deployed:
- **Mobile Apps / Dashcams**: Real-time road monitoring for municipal vehicles.
- **Municipal Dashboards**: Aggregating pothole locations, confidence, and size to schedule repairs.
- **Autonomous Driving**: Providing visual depth and obstacle detection for driverless cars to safely avoid hazards.

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **Ultralytics (YOLOv8n)**: Fine-tuned object detection
- **OpenCV & NumPy**: Image manipulation and scaling
- **Streamlit**: Premium web interface and visualization
- **Pillow**: Python imaging library
- **lxml**: XML parsing for Pascal VOC annotations
- **PyTorch**: Back-end GPU/CPU processing library

---

## 📁 Folder Structure

```text
road_pothole_detection/
├── app.py                     # Streamlit application
├── train.py                   # YOLOv8 training script
├── prepare_dataset.py         # Pascal VOC to YOLO converter
├── dataset/                   # Prepared dataset (YOLO format)
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   └── labels/
│       ├── train/
│       └── val/
├── models/
│   └── best.pt                # Fine-tuned best model weights
├── utils/
│   └── detection_utils.py     # Core machine learning & drawing helpers
├── data.yaml                  # YOLOv8 dataset config
├── requirements.txt           # Package dependencies
└── README.md                  # Project documentation
```

---

## 🚀 Setup & Usage Instructions

### Step 1: Install Dependencies
Open your command terminal and install the required Python libraries:
```bash
pip install -r requirements.txt
```

### Step 2: Download the Dataset
1. Download the pothole detection dataset from Kaggle:
   👉 [Kaggle Pothole Detection Dataset](https://www.kaggle.com/datasets/andrewmvd/pothole-detection)
2. Extract the archive. The dataset contains:
   - `images/` containing road pictures.
   - `annotations/` containing VOC XML files.
3. Place this dataset inside a folder named `raw_dataset/` inside the `road_pothole_detection/` directory:
   - `raw_dataset/images/`
   - `raw_dataset/annotations/`

*(Note: The system contains fallback configurations that will auto-link the folder if run inside the local pair-programming workspace.)*

### Step 3: Run Dataset Preparation
Run the script to parse XML files, convert coordinates to YOLO normalized format, filter out empty annotations, and split files (80% train, 20% validation):
```bash
python prepare_dataset.py
```

### Step 4: Train the YOLOv8 Model
Train the network on the prepared pothole labels:
```bash
python train.py
```
This runs fine-tuning on YOLOv8n for 50 epochs (with auto-patience of 15 epochs) and saves the best model output to `models/best.pt`.

### Step 5: Launch the Streamlit Web Application
Launch the visual web dashboard to upload images and analyze road surfaces:
```bash
streamlit run app.py
```

---

## ⚙️ How It Works (Metrics & Thresholds)

### Detection Thresholds
- **Confidence Threshold**: The minimum probability score required to classify a detected area as a pothole. If set to `0.25`, the model only highlights objects it is at least 25% sure are potholes.
- **IOU (Intersection over Union) Threshold**: Used during Non-Maximum Suppression (NMS) to eliminate duplicate overlapping bounding boxes around the same pothole. A lower value removes more overlapping boxes.

### Severity Levels
The system rates road damage severity based on the number of potholes detected:
- **0 potholes**: `None` — Road condition is good.
- **1-2 potholes**: `Low` — Minor damage, monitor road condition.
- **3-5 potholes**: `Medium` — Moderate damage, maintenance recommended.
- **6+ potholes**: `High` — Severe damage, immediate repair required!

### Coverage Area (%)
Calculated as the total area of all bounding boxes divided by the total area of the image, multiplied by 100. This indicates the percentage of the visible road surface occupied by potholes.

---

## 🛠️ Common Issues & Fixes

1. **Model Not Found at `models/best.pt`**:
   - Make sure you ran `python train.py` and that the training finished successfully.
   - Check if `best.pt` exists at `runs/pothole_detector/weights/best.pt` and has been copied over.
2. **No Detections Shown**:
   - Adjust the **Confidence Threshold** slider in the sidebar downwards (e.g., to `0.15` or `0.20`) to capture less distinct potholes.
3. **Out of Memory (CUDA OOM) / PyTorch Crashes**:
   - If training crashes due to GPU memory limits, modify the `batch` parameter in `train.py` from `16` to `8` or `4`.
4. **XML Parse Error / Directory Missing**:
   - Double-check your dataset path. The image files and annotations must be inside `raw_dataset/images/` and `raw_dataset/annotations/` respectively.
