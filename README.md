
# ECG Arrhythmia Detection with a Hybrid Prophet–MLP Pipeline

A feature-based machine learning pipeline that classifies ECG heartbeats as **normal** or **abnormal**. It combines statistical features, Fast Fourier Transform (FFT) spectral features, and Prophet time-series residuals, and classifies them with a tuned Multi-Layer Perceptron (MLP).

**Authors:** Mahtab Alikhani Najafabadi, Jamal Zarepour Ahmadabadi (Yazd University)

> **Status:** Research project (bachelor's thesis, graded 20/20). A manuscript is in preparation.

---

## Highlights

- End-to-end pipeline: filtering, feature extraction, scaling, class-imbalance handling, hyperparameter search, and evaluation.
- Three complementary views of each heartbeat: statistics, frequency content, and Prophet residuals.
- Tuned with 5-fold cross-validation (`GridSearchCV`) and balanced class weights.
- Evaluated with accuracy, precision, recall, F1, ROC-AUC, and a confusion matrix.

## Results

Evaluated on a held-out test set of **21,892 heartbeats** (18,118 normal, 3,774 abnormal).

| Metric   | Value  |
|----------|--------|
| Accuracy | 99.91% |
| ROC-AUC  | 0.9979 |

| Class    | Precision | Recall | F1-score |
|----------|-----------|--------|----------|
| Normal   | 0.9992    | 0.9996 | 0.9994   |
| Abnormal | 0.9984    | 0.9962 | 0.9973   |

Confusion matrix (rows: true class, columns: predicted class):

|                   | Predicted normal | Predicted abnormal |
|-------------------|------------------|--------------------|
| **True normal**   | 18,112           | 6                  |
| **True abnormal** | 14               | 3,760              |

![Confusion matrix](figures/confusion_matrix.png)

Best configuration found by grid search: one hidden layer with 100 neurons, learning rate 0.001.

> **Please read the Limitations section before interpreting these numbers.**

## Dataset

- **Source:** heartbeat segments derived from the **MIT-BIH Arrhythmia Database**, in the pre-processed CSV format available on Kaggle (*ECG Heartbeat Categorization Dataset*, built by Kachuee et al., 2018).
- **Files used:** `mitbih_train.csv` and `mitbih_test.csv`.
- Each row holds **187 time samples** of one heartbeat plus a class label in the last column.
- Labels are converted to binary: `0` = normal, `1` = abnormal (all non-normal classes merged).
- The data is **not included** in this repository due to its size. Download it and place the two CSV files in the `data/` folder.

## Method

1. Load the training and test data.
2. Reduce noise with a 0.5–40 Hz band-pass filter (SciPy).
3. Extract statistical features per heartbeat: mean, standard deviation, median, skewness, kurtosis.
4. Compute Prophet residuals (observed minus forecast) as additional features.
5. Compute FFT-based frequency-domain features.
6. Combine all features and standardize them with `StandardScaler`.
7. Handle class imbalance with balanced class weights.
8. Tune an `MLPClassifier` with `GridSearchCV` (5-fold cross-validation) over hidden-layer sizes and learning rates.
9. Evaluate on the held-out test set.

The complete implementation of steps 1–9 is in **`mlp&prophet_improved.py`**.

## Limitations

This section is deliberately explicit, because it matters for how the results should be read.

- **Beat-level split, not patient-level.** The provided train/test files are split at the heartbeat level, so beats from the same patient can appear in both sets. Accuracy on unseen patients is therefore likely lower than the figures above. An inter-patient evaluation (for example the DS1/DS2 protocol of de Chazal et al.) is needed to measure true generalization.
- **Binary task.** All abnormal beat types are merged into one class. The results are not directly comparable with multi-class studies.
- **Single dataset.** No external validation on other ECG databases has been performed.
- **Component contribution is untested.** No ablation study yet shows how much Prophet residuals and FFT features each add compared with statistical features alone.

## Future Work

- Inter-patient evaluation and per-class sensitivity reporting.
- Ablation study (statistical features only, +FFT, +Prophet, all).
- Comparison with 1D-CNN and LSTM baselines under the same protocol.
- Multi-class classification of individual arrhythmia types.

## Project Structure

```
ECG-Arrhythmia-Classification/
├── data/                       # place mitbih_train.csv and mitbih_test.csv here (not tracked)
├── figures/                    # confusion matrix and other plots
├── mlp&prophet_improved.py     # main pipeline (filtering → features → tuning → evaluation)
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

**Requirements:** Python 3.9+

```bash
git clone https://github.com/Mahtab-alikhani/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
pip install -r requirements.txt
```

Download the dataset (see above), place the CSV files in `data/`, then run the main script:

```bash
python "mlp&prophet_improved.py"
```

`requirements.txt`:

```
pandas
numpy
scipy
scikit-learn
prophet
matplotlib
seaborn
```

## Reproducibility

- Random seeds are fixed inside `mlp&prophet_improved.py` for the train/test handling, `GridSearchCV`, and the MLP.
- The grid-search configuration reported in the Results section corresponds to `hidden_layer_sizes = [(100,)]` and `learning_rate_init = 0.001`.
- Running the script on the same `mitbih_train.csv` / `mitbih_test.csv` should reproduce the reported confusion matrix (small variations may occur across library versions).

## Citation

If you use this work, please cite the dataset sources:

- Moody, G. B., & Mark, R. G. (2001). The impact of the MIT-BIH Arrhythmia Database. *IEEE Engineering in Medicine and Biology Magazine*, 20(3), 45–50.
- Goldberger, A. L., et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet. *Circulation*, 101(23), e215–e220.
- Kachuee, M., Fazeli, S., & Sarrafzadeh, M. (2018). ECG heartbeat classification: a deep transferable representation. *IEEE ICHI 2018*.

## License

Released under the MIT License. See `LICENSE`.

## Contact

Mahtab Alikhani Najafabadi — mahtab.alikhani.nj@gmail.com
```
## Contact
Mahtab Alikhani Najafabadi: mahtab.alikhani.nj@gmail.com
