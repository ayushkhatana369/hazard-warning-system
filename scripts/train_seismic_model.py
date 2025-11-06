import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Load preprocessed normalized seismic waveform data
data_norm = np.load('data/seismic/seismic_normalized.npy')

# For demonstration, let's create synthetic binary labels:
segment_length = 1000
num_segments = len(data_norm) // segment_length
X = np.array([data_norm[i*segment_length:(i+1)*segment_length] for i in range(num_segments)])

# Synthetic binary labels for classification (simulate earthquake vs no-earthquake)
y = np.array([1 if i % 2 == 0 else 0 for i in range(num_segments)])

# Split train and test datasets (80% train)
split_idx = int(0.8 * num_segments)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for model input (samples, features)
X_train = X_train.reshape(-1, segment_length)
X_test = X_test.reshape(-1, segment_length)

# Initialize Feedforward NN using tf.keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(segment_length,)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Binary classifier
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test))

# Plot training history
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model for future use
model.save('models/seismic_classifier.h5')
