# diabetic-retinopathy-detection
Deep learning-based diabetic retinopathy severity grading using retinal fundus images, transfer learning, model evaluation, and explainable AI.

# Diabetic Retinopathy Detection Using Deep Learning


## Overview

This project aims to develop a deep learning pipeline for grading diabetic retinopathy severity from retinal fundus images using the APTOS 2019 Blindness Detection dataset.

The goal is to compare convolutional neural network architectures, evaluate classification performance, investigate class imbalance, and interpret model predictions using Grad-CAM.

## Objectives

- Perform exploratory data analysis and retinal image preprocessing.
- Develop a baseline image classification model using PyTorch.
- Compare pretrained CNN architectures such as ResNet50, EfficientNet, and InceptionV3.
- Evaluate models using accuracy, macro F1-score, confusion matrices, and quadratic weighted kappa.
- Investigate methods for improving performance on underrepresented severity classes.
- Visualize model predictions using Grad-CAM.
- Build a simple interface for testing the trained model.

## Planned Tech Stack

- Python
- PyTorch and Torchvision
- NumPy and Pandas
- OpenCV
- Scikit-learn
- Matplotlib
- Grad-CAM
- Google Colab

## Dataset

APTOS 2019 Blindness Detection — Kaggle.

Dataset access: https://www.kaggle.com/competitions/aptos2019-blindness-detection

The dataset contains retinal fundus images labeled with five diabetic retinopathy severity grades, from 0 (No DR) to 4 (Proliferative DR).

Dataset images and labels will be handled according to the applicable dataset terms.

## Project Structure

```text
diabetic-retinopathy-detection/
├── notebooks/
├── src/
├── outputs/
├── requirements.txt
├── .gitignore
└── README.md
```


## Disclaimer

This is an educational machine learning project, not a clinically validated diagnostic system. Model predictions must not be used as a substitute for professional medical evaluation.
