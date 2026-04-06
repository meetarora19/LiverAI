# LiverAI CT Pipeline — v1.0

This project implements a CT-based liver analysis pipeline using the Task03_Liver dataset from the Medical Segmentation Decathlon.

## Features
- 3D CT dataset inspection
- 3D → 2D slice extraction
- Normal vs Tumor slice classification
- Train / validation / test split generation
- CNN-based classifier (MobileNetV2)
- Saved training plots and evaluation metrics

## Project Structure
src/ct_pipeline/ → core scripts  
data/ct_liver/processed/ → generated slices and splits  
data/ct_liver/outputs/ → metrics and visualizations  
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

## Outputs
- Accuracy and loss plots (PNG)
- Confusion matrix
- Classification report
- Saved trained model

## Notes
- Raw dataset is not included due to size (~26GB)
- This is a baseline CT classification pipeline
- Future versions will include prediction and morphology analysis