import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils import class_weight
from prophet import Prophet

# --------------------------
# 1. Load dataset
# --------------------------
train_data = pd.read_csv('mitbih_train.csv', header=None)
test_data = pd.read_csv('mitbih_test.csv', header=None)

# --------------------------
# 2. Compute features
# --------------------------
def compute_features(data):
    mean_vals = data.mean(axis=1)
    std_vals = data.std(axis=1)
    median_vals = data.median(axis=1)
    skew_vals = data.apply(skew, axis=1)
    kurt_vals = data.apply(kurtosis, axis=1)
    return np.vstack([mean_vals, std_vals, median_vals, skew_vals, kurt_vals]).T

train_features = compute_features(train_data.iloc[:, :-1])
test_features = compute_features(test_data.iloc[:, :-1])

# --------------------------
# 3. Prophet residuals
# --------------------------
train_dates = pd.date_range(start='2023-01-01', periods=len(train_features), freq='S')
train_df = pd.DataFrame({'ds': train_dates, 'y': train_features[:,0]})
model = Prophet(yearly_seasonality=True, weekly_seasonality=True, changepoint_prior_scale=0.05)
model.fit(train_df)

future = model.make_future_dataframe(periods=len(test_features))
forecast = model.predict(future)
residuals = test_features[:,0] - forecast['yhat'][-len(test_features):].values
residuals = residuals.reshape(-1,1)

# --------------------------
# 4. FFT features
# --------------------------
def fft_features(data, n_fft=10):
    padded = np.pad(data, (0, n_fft - len(data)%n_fft if len(data)%n_fft!=0 else 0), 'constant')
    fft_vals = np.abs(fft(padded))
    return fft_vals[:len(data)]

fft_feat = np.array([fft_features([v], n_fft=5) for v in test_features[:,0]])

# --------------------------
# 5. Combine all features
# --------------------------
X = np.hstack([test_features, residuals, fft_feat])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = (test_data.iloc[:, -1].values != 0).astype(int)

# --------------------------
# 6. Class weights
# --------------------------
weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i:w for i,w in enumerate(weights)}

# --------------------------
# 7. MLP with GridSearchCV
# --------------------------
mlp = MLPClassifier(random_state=42, max_iter=2000)
param_grid = {
    'hidden_layer_sizes': [(100,), (100,50), (200,100)],
    'learning_rate_init': [0.001, 0.005]
}
cv = KFold(n_splits=5, shuffle=True, random_state=42)
grid_search = GridSearchCV(mlp, param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_scaled, y)

best_mlp = grid_search.best_estimator_
print(f"Best MLP parameters: {grid_search.best_params_}\n")

# --------------------------
# 8. Predictions
# --------------------------
y_pred = best_mlp.predict(X_scaled)

# --------------------------
# 9. Normal Classification Report
# --------------------------
print("Classification Report:")
print(classification_report(y, y_pred, labels=[0,1], target_names=['Normal','Anomalous'], zero_division=0))

# --------------------------
# 10. Custom Report (2 رقم اعشار بدون گرد کردن)
# --------------------------
precisions = precision_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
recalls = recall_score(y, y_pred, labels=[0,1], average=None, zero_division=0)
f1s = f1_score(y, y_pred, labels=[0,1], average=None, zero_division=0)

def format_truncate(val):
    s = f"{val:.6f}"   # تا 4 رقم اعشار تولید می‌کنیم
    return s[:s.find('.')+5]  # فقط 2 رقم بعد اعشار، بدون گرد کردن

print("\nCustom Classification Report (4 رقم اعشار بدون گرد کردن):\n")
print(f"{'Class':<12}{'Precision':<12}{'Recall':<12}{'F1-score':<12}")
for i, cls in enumerate(['Normal','Anomalous']):
    print(f"{cls:<12}{format_truncate(precisions[i]):<12}{format_truncate(recalls[i]):<12}{format_truncate(f1s[i]):<12}")

acc = accuracy_score(y, y_pred)
roc = roc_auc_score(y, y_pred)
print(f"\nAccuracy: {format_truncate(acc*100)}%")
print(f"ROC-AUC Score: {format_truncate(roc)}")

# --------------------------
# 11. Confusion Matrix
# --------------------------
cm = confusion_matrix(y, y_pred, labels=[0,1])
print("\nConfusion Matrix (Counts):")
print(cm)

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print("\nConfusion Matrix (Normalized %):")
print(np.round(cm_normalized*100,2))