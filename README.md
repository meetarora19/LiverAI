# LiverAI Pipeline — v1.2

This project implements a unified liver analysis system combining CT-based tumor detection and ultrasound-based fatty liver classification.

## Features
- CT scan tumor detection (slice-based)
- Fatty liver classification (ultrasound-based)
- Combined prediction pipeline
- Morphology analysis (tumor size, liver area, ratios)
- Saved prediction outputs (PNG + JSON)

## Project Structure
src/ct_pipeline/ → CT pipeline scripts  
src/fatty_pipeline/ → fatty liver training notebook  
src/combined/ → unified prediction system  
data/ct_liver/processed/ → generated slices and masks  
data/ct_liver/outputs/ → predictions and metrics  
models/ → trained models  

## Models
- CT Model → Normal vs Tumor classification  
- Fatty Liver Model → NAFLD vs Non-NAFLD  

## Dataset
The CT dataset used is **Task03_Liver** from the Medical Segmentation Decathlon.

Download from:
https://medicaldecathlon.com/

After downloading, place it in:

data/ct_liver/raw/Task03_Liver/

## Setup
1. Create environment:
   py -3.11 -m venv venv  
   venv\Scripts\activate  

2. Install dependencies:
   python -m pip install -r requirements.txt  

## Combined Prediction
python src/combined/combined_predict.py <image_path>

Outputs:
- Combined prediction image
- JSON report including:
  - Fatty liver result
  - CT tumor result
  - Morphology metrics (if available)
  - Overall health status

## Outputs
- Prediction visualizations (PNG)
- Structured JSON results
- Morphology measurements

## Notes
- Both models use standardized input size (224x224)
- CT and ultrasound modalities are processed independently and combined at output level
- Morphology analysis is available only when corresponding mask files exist
- Raw dataset (~26GB) is not included in the repository