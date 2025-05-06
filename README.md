
# Image-Based Facial Emotion Recognition

![Python](https://img.shields.io/badge/Python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 📌 Project Overview

This project presents an image-based facial emotion recognition system using deep learning, with a comparative analysis against traditional machine learning approaches. It aims to classify human facial expressions into **seven emotion categories** using the **FER-2013 dataset**, handling challenges like **class imbalance** and **real-time performance**.

## ✨ Key Features

- ✅ Deep learning with CNNs for emotion classification
- ✅ Traditional ML baselines: k-NN and SVM with HOG+LBP features
- ✅ Focal Loss and Class-Balanced Batches for minority class improvement
- ✅ Data augmentation for generalisation
- ✅ Real-time emotion detection using OpenCV and webcam
- ✅ Evaluation with accuracy, precision, recall, F1-score, and confusion matrix

## 📂 Emotion Classes
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

## 🖥️ Real-Time System
The trained CNN is deployed with **OpenCV** to detect and classify emotions via webcam in real-time.  
**Platform:** macOS (AVFoundation compatible)  
**Input:** Grayscale face image (48x48)

## 📁 Dataset
FER-2013 (Kaggle):  
[https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/facial-emotion-recognition.git
cd facial-emotion-recognition
```

### 2. Install Requirements
```bash
pip install -r requirements.txt
```

### 3. Train or Load Model
```python
# Train
python train_cnn.py

# Real-time Prediction
python webcam_emotion.py
```

## 📚 References
Key references include:
- FER-2013 Dataset
- CNN Architectures
- LBP & HOG feature descriptors

> For full references, please refer to the [Project Report](./Group_10_Project_Report.pdf)

## 👥 Contributors
- **Sujan Khanal** (u3258630@uni.canberra.edu.au)  
- **Rohit Baral** (u3268702@uni.canberra.edu.au)

