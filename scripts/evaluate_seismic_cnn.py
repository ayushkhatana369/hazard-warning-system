import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# Load best model
model = tf.keras.models.load_model('models/seismic_best_cnn.h5')

# Load spectrogram samples and labels
spec = np.load('data/seismic/spectrogram.npy')
window_size = 64
time_slices = spec.shape[1]

labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1

X_samples, y_samples = [], []
for start in range(0, time_slices - window_size):
    sample = spec[:, start:start + window_size]
    if sample.shape == (spec.shape[0], window_size):
        X_samples.append(sample)
        y_samples.append(labels[start + window_size // 2])
X_samples = np.array(X_samples)[..., np.newaxis]
y_samples = np.array(y_samples)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred.flatten() > 0.5).astype(int)

print("Accuracy:", accuracy_score(y_test, y_pred_binary))
print("Precision:", precision_score(y_test, y_pred_binary))
print("Recall:", recall_score(y_test, y_pred_binary))
print("F1 Score:", f1_score(y_test, y_pred_binary))
print("Classification Report:\n", classification_report(y_test, y_pred_binary))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_binary)
plt.matshow(cm, cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.colorbar()
for (i, j), val in np.ndenumerate(cm):
    plt.text(j, i, val, ha='center', va='center')
plt.show()
