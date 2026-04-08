# LiverAI CT Pipeline — v1.1

This project implements a CT-based liver analysis pipeline using the Task03_Liver dataset from the Medical Segmentation Decathlon.

## Features
- 3D CT dataset inspection
- 3D → 2D slice extraction
- Normal vs Tumor slice classification
- Train / validation / test split generation
- CNN-based classifier (MobileNetV2)
- Saved training plots and evaluation metrics
- Single-slice CT prediction utility
- Basic morphology analysis using liver and tumor masks

## Project Structure
src/ct_pipeline/ → core CT pipeline scripts  
data/ct_liver/processed/ → generated slices and masks  
data/ct_liver/outputs/ → metrics, predictions, and visualizations  
models/ct_model/ → trained classifier  

## Dataset
The dataset used is **Task03_Liver** from the Medical Segmentation Decathlon.

Download from:
https://medicaldecathlon.com/

After downloading, place it in:

data/ct_liver/raw/Task03_Liver/

Ensure structure:
Task03_Liver/
├── imagesTr/
├── labelsTr/
├── imagesTs/
└── dataset.json

## Setup
1. Create environment:
   py -3.11 -m venv venv  
   venv\Scripts\activate  

2. Install dependencies:
   python -m pip install -r requirements.txt  

## Run Pipeline
python src/ct_pipeline/inspect_ct_dataset.py  
python src/ct_pipeline/prepare_slices.py  
python src/ct_pipeline/create_splits.py  
python src/ct_pipeline/check_splits.py  
python src/ct_pipeline/train_classifier.py  

## Prediction
python src/ct_pipeline/predict_ct.py <image_path>

Outputs:
- Saved prediction image
- JSON result file

## Morphology Analysis
python src/ct_pipeline/morphology_analysis.py <slice_name>

Outputs:
- Liver area (pixels)
- Tumor area (pixels)
- Tumor-to-liver ratio
- Tumor dimensions (bounding box)

## Outputs
- Accuracy and loss plots (PNG)
- Confusion matrix
- Classification report
- Saved trained model
- Prediction images and JSON files
- Morphology analysis results

## Notes
- Raw dataset (~26GB) is not included
- This version extends the baseline CT classifier with prediction and morphology analysis