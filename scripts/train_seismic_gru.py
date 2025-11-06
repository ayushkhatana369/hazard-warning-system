import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load spectrogram
spec = np.load('data/seismic/spectrogram.npy')
window_size = 64
time_slices = spec.shape[1]

labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1

X_samples, y_samples = [], []
for start in range(0, time_slices - window_size):
    sample = spec[:, start:start + window_size].T
    if sample.shape == (window_size, spec.shape[0]):
        X_samples.append(sample)
        y_samples.append(labels[start + window_size // 2])

X_samples = np.array(X_samples)
y_samples = np.array(y_samples)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

# Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(window_size, spec.shape[0]), return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Plot training
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('GRU Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model
model.save('models/seismic_best_gru.h5')

