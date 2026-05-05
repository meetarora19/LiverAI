# LiverAI: Multi-Class Liver Abnormality Detection

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15.0-FF6F00.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8.0-green.svg)

LiverAI is a unified, multi-modal medical imaging pipeline designed to detect and analyze liver abnormalities with high computational efficiency. By combining CT-based structural tumor detection with ultrasound-based fatty liver (NAFLD) classification, LiverAI provides a holistic, automated triage system for clinical decision support.

## Key Features

* **Dual-Modality Architecture:** Processes both 3D CT volumes (converted to 2D slices) and 2D ultrasound imagery.
* **Lightweight Backbone:** Utilizes a custom-tuned MobileNetV2 architecture for both pipelines, drastically reducing parameter count for fast, real-time inference without sacrificing accuracy.
* **Morphology Analysis Engine:** Uses OpenCV and segmentation masks to calculate precise liver area, tumor area, and critical lesion-to-liver ratios, moving beyond simple binary predictions.
* **Uncertainty Margin:** Built-in confidence thresholding flags ambiguous scans as "Uncertain," ensuring safe clinical AI deployment.
* **Unified Diagnostic Export:** Aggregates findings into a single timestamped JSON payload and a fully labeled visual prediction map.

## Repository Structure

```text
LiverAI/
├── data/
│   └── ct_liver/
│       ├── outputs/        # Generated JSON reports and PNG visual predictions
│       ├── processed/      # Extracted 2D slices and corresponding masks
│       └── raw/            # (Ignored) Task03_Liver raw .nii.gz dataset
├── models/
│   ├── ct_model/           # Saved MobileNetV2 tumor classifier (.keras)
│   └── fatty_liver/        # Saved fine-tuned NAFLD classifier (.keras)
├── src/
│   ├── combined/
│   │   └── combined_predict.py   # Master orchestration pipeline
│   ├── ct_pipeline/
│   │   ├── create_splits.py      # Automated train/val/test splitter
│   │   ├── morphology_analysis.py# Structural feature extraction 
│   │   ├── predict_ct.py         # CT inference script
│   │   ├── prepare_slices.py     # 3D to 2D volume processor
│   │   └── train_classifier.py   # Model training logic
│   └── fatty_pipeline/
│       └── fatty_liver_model.ipynb # Two-stage transfer learning notebook
├── .gitignore
├── README.md
└── requirements.txt
```

## Datasets

1. **CT Tumor Dataset:** Medical Segmentation Decathlon (Task03_Liver). Place the downloaded `imagesTr` and `labelsTr` folders into `data/ct_liver/raw/Task03_Liver/`.
2. **Fatty Liver Dataset:** Ultrasound imagery optimized for binary classification of NAFLD vs. Non-NAFLD.

*(Note: Raw volumetric datasets are excluded from this repository due to size constraints.)*

## Setup & Installation

**1. Clone the repository:**
```bash
git clone [https://github.com/meetarora19/LiverAI.git](https://github.com/meetarora19/LiverAI.git)
cd LiverAI
```

**2. Create and activate a virtual environment (Python 3.11+ recommended):**
```bash
py -3.11 -m venv venv  
venv\Scripts\activate  # On macOS/Linux use: source venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

To run the unified aggregator and process a medical image through the complete LiverAI pipeline:

```bash
python src/combined/combined_predict.py <path_to_image>
```

**Outputs:**
The script will generate a timestamped file pair in `data/ct_liver/outputs/predictions/`:
1. `prediction_timestamp.png`: The original image overlaid with the final diagnostic result, predicted type, and confidence scores.
2. `prediction_timestamp.json`: A structured clinical payload containing raw probabilities, morphology metrics (liver area, tumor area, ratio), and the final triaged decision.

## Developer

**Meet Arora**
*Computer Science Undergraduate | AI & ML, Big Data & Analytics*
Bennett University

* **GitHub:** [meetarora19](https://github.com/meetarora19)
* **LinkedIn:** [arora-meet](www.linkedin.com/in/arora-meet)
* **Email:** meetarora19.005@gmail.com