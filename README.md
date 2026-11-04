# ECG-Arrhythmia-Classification
A hybrid Prophet-MLP architecture for ECG Arrhythmia classification achieving 99.91% accuracy.
A hybrid machine learning approach for classifying ECG signals into **Normal** and **Anomalous** classes using statistical features, Prophet time-series residuals, FFT spectral features, and an optimized Multi-Layer Perceptron (MLP).

**Authors:** Mahtab Alikhani-Najafabadi, Jamal Zarepour-Ahmadabadi

## Overview

This project implements a hybrid framework for binary ECG arrhythmia detection on the **MIT-BIH Arrhythmia Database**. The method combines:

- Statistical features (mean, standard deviation, median, skewness, kurtosis)
- Prophet time-series modeling for residual extraction
- Fast Fourier Transform (FFT) for frequency-domain features
- Optimized Multi-Layer Perceptron (MLP) classifier with GridSearchCV

### Key Results
- **Accuracy:** 99.91%
- **ROC-AUC:** 0.9979
- Extremely low false positive and false negative rates

## Dataset
- **MIT-BIH Arrhythmia Database**
- Files used: `mitbih_train.csv` and `mitbih_test.csv`
- Each row contains 187 time samples + 1 label
- Labels are converted to binary: `0 = Normal`, `1 = Anomalous`

> **Note:** The dataset is not included in this repository due to size.  
> Download it from [PhysioNet MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) or Kaggle.

## Method Pipeline

1. Load train and test data
2. Extract statistical features (mean, std, median, skewness, kurtosis)
3. Compute Prophet residuals
4. Extract FFT features
5. Combine all features and apply StandardScaler
6. Handle class imbalance using balanced class weights
7. Optimize MLP using GridSearchCV with 5-fold cross-validation
8. Evaluate performance (Accuracy, Precision, Recall, F1-score, ROC-AUC, Confusion Matrix)

## Project Structure

## Method Pipeline
1. Noise reduction using bandpass filter (0.5–40 Hz)
2. Statistical feature extraction
3. FFT-based frequency features
4. Prophet residual calculation
5. Feature standardization
6. MLP classification (best configuration: single hidden layer with 100 neurons, learning rate = 0.001)

## Project StructureBIH dataset, the raw data is not included. Please download it from PhysioNet.Results Sum

Email: [mahtab.alikhani.nj@gmail.com]LicenseThis project is for academic and research purposes.text---

requirements.txt content:textpandas
numpy
scipy
scikit-learn
prophet
matplotlib
seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```

## Requirements

Install the required libraries:

```bash
pip install -r requirements.txt
```
