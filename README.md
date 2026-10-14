# Diabetic Retinopathy Classification Using Deep Learning

An experimental deep learning project for five-class diabetic retinopathy classification from retinal fundus images.

## Overview
- Dataset: APTOS 2019 Blindness Detection
- Framework: PyTorch
- Models: ResNet50, Class-Weighted ResNet50, EfficientNet-B0
- Explainability: Grad-CAM
- Demo: Gradio

## Severity Grades
- Grade 0: No DR
- Grade 1: Mild
- Grade 2: Moderate
- Grade 3: Severe
- Grade 4: Proliferative DR

## Methodology
1. Create a stratified 80/20 train-validation split using random state 42.
2. Resize images to 224x224 and normalize using ImageNet statistics.
3. Fine-tune pretrained CNN architectures.
4. Evaluate using accuracy, macro F1-score, and quadratic weighted kappa.
5. Use Grad-CAM to visualize influential image regions.

## Validation Results
| Model | Accuracy | Macro F1 | Quadratic Weighted Kappa |
|---|---:|---:|---:|
| ResNet50 | 0.8063 | 0.6385 | 0.8659 |
| Class-Weighted ResNet50 | 0.8104 | 0.6381 | 0.8481 |
| EfficientNet-B0 | 0.8104 | 0.6490 | 0.8742 |

EfficientNet-B0 achieved the highest macro F1-score and quadratic weighted kappa among these evaluated checkpoints.

## Repository Structure
- src/: inference and application code
- results/: evaluation metrics
- visualizations/: Grad-CAM images
- docs/: additional documentation

## Setup
Install dependencies with: pip install torch torchvision pandas numpy pillow matplotlib scikit-learn opencv-python-headless gradio

Download the APTOS 2019 dataset from Kaggle separately. Dataset images and model checkpoints are not included in this repository.

## Limitations
- Evaluation uses a single stratified validation split.
- Minority-grade performance remains challenging.
- Softmax scores are not calibrated clinical probabilities.
- Grad-CAM does not prove medical reasoning.
- Independent external clinical validation has not been performed.

## Intended Use
For educational and research purposes only. This project is not intended for clinical diagnosis or treatment decisions.
