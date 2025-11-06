import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

models = {
    "CNN": 'models/seismic_best_cnn.h5',
    "LSTM": 'models/seismic_best_lstm.h5',
    "GRU": 'models/seismic_best_gru.h5'
}

# Generate your test data
spec = np.load('data/seismic/spectrogram.npy')
window_size = 64
time_slices = spec.shape[1]
labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1
X_samples, y_samples = [], []
for start in range(0, time_slices - window_size):
    # Choose LSTM/GRU format or CNN as needed; here is for CNN
    sample = spec[:, start:start + window_size]
    if sample.shape == (spec.shape[0], window_size):
        X_samples.append(sample)
        y_samples.append(labels[start + window_size // 2])
X_samples = np.array(X_samples)[..., np.newaxis]
y_samples = np.array(y_samples)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

# Evaluate all models
results = []
for model_name, model_path in models.items():
    model = tf.keras.models.load_model(model_path)
    if "CNN" in model_name:
        # CNN expects (freq, window, 1)
        eval_X = X_test
    else:
        # LSTM/GRU expects (window, freq)
        eval_X = X_test.squeeze().transpose(0,2,1) if X_test.ndim == 4 else X_test
    y_pred = model.predict(eval_X)
    y_pred_binary = (y_pred.flatten() > 0.5).astype(int)
    results.append({
        "Model": model_name,
        "Accuracy": accuracy_score(y_test, y_pred_binary),
        "Precision": precision_score(y_test, y_pred_binary),
        "Recall": recall_score(y_test, y_pred_binary),
        "F1 Score": f1_score(y_test, y_pred_binary)
    })

# Print comparison table
print("Model Performance Comparison:")
print("{:<8} {:<10} {:<10} {:<10} {:<10}".format("Model","Accuracy","Precision","Recall","F1 Score"))
for r in results:
    print("{:<8} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(r["Model"],r["Accuracy"],r["Precision"],r["Recall"],r["F1 Score"]))

