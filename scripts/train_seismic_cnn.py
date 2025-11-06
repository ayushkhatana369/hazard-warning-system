import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load spectrogram
spec = np.load('data/seismic/spectrogram.npy')

print('Spectrogram shape:', spec.shape)  # (frequencies, time_slices)

window_size = 64
time_slices = spec.shape[1]

X_samples = []
y_samples = []

# Labels example: first half earthquake (1), second half no event (0)
labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1

# Generate samples using sliding window approach with valid shape check
for start in range(0, time_slices - window_size):
    sample = spec[:, start:start + window_size]
    if sample.shape == (spec.shape[0], window_size):
        X_samples.append(sample)
        y_samples.append(labels[start + window_size // 2])

X_samples = np.array(X_samples)
y_samples = np.array(y_samples)

print('Number of samples:', len(X_samples))

# Add channel dimension for CNN
X_samples = X_samples[..., np.newaxis]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

# Build CNN model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=X_train.shape[1:]),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Plot training history
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model
model.save('models/seismic_cnn_classifier.h5')
