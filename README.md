# Micro-Expression Lie Detection using Machine Learning

This project implements a machine learning pipeline for classifying **micro-expressions** into **lie** or **truth** using extracted image-based features from facial landmarks. The classification models used include:

- Support Vector Machine (SVM)
- Random Forest Classifier (RF)
- K-Nearest Neighbors (KNN)

## 🧠 Objective

To detect deceptive behavior by analyzing micro facial expressions using image feature extraction and classic ML algorithms.

---

## 📁 Dataset

We use a publicly available micro-expression dataset for lie detection:

- **Kaggle**: [Micro-Expression Dataset for Lie Detection](https://www.kaggle.com/datasets/devvratmathur/micro-expression-dataset-for-lie-detection)
- **GitHub**: [achelousace/Micro-Facial-Expression-Detection](https://github.com/achelousace/Micro-Facial-Expression-Detection)

From the raw images, we extract facial landmarks and compute top 8 engineered features, saved in:

engineered_features_top8.csv

Each data point is labeled:
- `0` → Lie
- `1` → Truth

---

## 🛠️ Features

- Facial landmark extraction
- Feature engineering on facial distances/ratios
- Binary classification (Lie vs Truth)
- Visualization using `OpenCV` and `MediaPipe`
- Probability prediction with trained models

---

## 🧪 Training Scripts

We provide training scripts for:

### ✅ Support Vector Machine (SVM)
- Trained on `engineered_features_top8.csv`
- Model saved as `svm.pkl`

### 🌲 Random Forest
- Trained and saved as `randomforest.pkl`
- Includes accuracy, classification report, and confusion matrix

### 🔍 K-Nearest Neighbors (KNN)
- Trained with `k=5` and saved as `knn.pkl`
- Outputs class probabilities on prediction

All models use stratified train-test split and report evaluation metrics.

---

## 🎮 Demo Script

- Loads trained model and `scaler.pkl`
- Takes image input
- Predicts `Lie` or `Truth`
- Overlays results with clean landmark visualizations

---

## 📦 Dependencies

- Python 3.x
- `pandas`, `numpy`, `scikit-learn`
- `opencv-python`, `mediapipe`
- `matplotlib`, `seaborn`, `joblib`

Install all dependencies with:

```bash
pip install -r requirements.txt

