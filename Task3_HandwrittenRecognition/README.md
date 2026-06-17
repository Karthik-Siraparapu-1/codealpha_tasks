# Task 3 - Handwritten Character Recognition

## Objective
Build an AI model to recognize handwritten digits using deep learning.

## Project Overview
This project uses the MNIST dataset to train a Convolutional Neural Network (CNN) for handwritten digit recognition. The model classifies handwritten digits from 0 to 9 with high accuracy.

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Seaborn
- Joblib

## Features
- Automatic MNIST dataset loading
- Image preprocessing and normalization
- CNN model training
- Accuracy and loss visualization
- Confusion matrix generation
- Sample handwritten digit prediction
- Model saving for future use
- Performance report generation

## Workflow
1. Load MNIST dataset
2. Normalize image pixels
3. Train CNN model
4. Evaluate model performance
5. Generate prediction outputs
6. Save graphs and reports

## Model Performance
- Final Accuracy: ~98% to 99%
- High precision and recall across all digit classes

## Outputs

Stored in `outputs/`:
- confusion_matrix.png
- training_accuracy.png
- training_loss.png
- sample_prediction.png

Stored in `models/`:
- handwritten_model.keras
- model_report.pkl

## How to Run

```bash
python3 handwritten_recognition.py