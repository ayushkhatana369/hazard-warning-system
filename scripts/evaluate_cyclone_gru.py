import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Load model
model = tf.keras.models.load_model('models/cyclone_best_gru.h5')

# Load test data (use same loading & preprocessing as training)
time_slices = 1000
window_size = 64
num_features = 6

data = np.random.rand(time_slices, num_features)  # Replace with actual test features
labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1

X_samples, y_samples = [], []
for start in range(time_slices - window_size):
    segment = data[start:start + window_size, :]
    label = labels[start + window_size // 2]
    X_samples.append(segment)
    y_samples.append(label)

X_samples = np.array(X_samples)
y_samples = np.array(y_samples)

# Split test data
from sklearn.model_selection import train_test_split
_, X_test, _, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

# Predict
y_pred_proba = model.predict(X_test)
y_pred = (y_pred_proba.flatten() > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
