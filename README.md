# ECG-Arrhythmia-Classification
A hybrid Prophet-MLP architecture for ECG Arrhythmia classification achieving 99.91% accuracy.
# Hybrid Prophet-MLP for ECG Signal Classification

**Authors:** Mahtab Alikhani-Najafabadi, Jamal Zarepour-Ahmadabadi

## Overview

This project presents a hybrid framework for binary classification of ECG signals (Normal vs Abnormal) evaluated on the **MIT-BIH Arrhythmia Database**.

The pipeline integrates:
- Statistical features (mean, std, median, skewness, kurtosis)
- Fast Fourier Transform (FFT) for spectral features
- Prophet time-series model for residual extraction
- Principal Component Analysis (PCA) for dimensionality reduction
- Optimized Multi-Layer Perceptron (MLP) classifier

### Key Results
- **Accuracy:** 99.91%
- **ROC-AUC:** 0.9979
- **Test samples:** 21,892
- Extremely low false positive and false negative rates

## Dataset
- **MIT-BIH Arrhythmia Database** (PhysioNet)
- Sampling frequency: 360 Hz
- Binary labels: Normal (0) / Abnormal (1)

## Method Pipeline
1. Noise reduction using bandpass filter (0.5–40 Hz)
2. Statistical feature extraction
3. FFT-based frequency features
4. Prophet residual calculation
5. Feature standardization
6. MLP classification (best configuration: single hidden layer with 100 neurons, learning rate = 0.001)

## Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- scipy
- prophet (Facebook Prophet)
- matplotlib / seaborn

Install dependencies:
```bash
pip install -r requirements.txt
```
How to Run
Clone the repository:
Bashgit clone https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
cd YOUR-REPO-NAME
Install requirements:
Bashpip install -r requirements.txt
Run the main script or notebook (update paths according to your data location).
Note: Due to the size of the MIT-BIH dataset, the raw data is not included. Please download it from PhysioNet.Results Summary

MetricValueAccuracy99.91%ROC-AUC0.9979Precision (Abnormal)0.9984Recall (Abnormal)0.9962F1-score (Abnormal)0.9973CitationIf you use this work, please cite:Alikhani-Najafabadi, M., & Zarepour-Ahmadabadi, J. (2025). A Hybrid Prophet-MLP Approach for ECG Signal Classification.ContactMahtab Alikhani-Najafabadi

Email: [mahtab.alikhani.nj@gmail.com]LicenseThis project is for academic and research purposes.text---
