import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load or simulate cyclone multivariate time series data
# Replace with actual data loading logic
time_slices = 1000
window_size = 64
num_features = 6  # e.g., lat, lon, pressure, wind speed, storm age, distance to land

data = np.random.rand(time_slices, num_features)  # Replace with actual cyclone feature matrix

# Example binary labels for classification (modify as needed)
labels = np.zeros(time_slices)
labels[:time_slices // 2] = 1

# Prepare time windows
X_samples, y_samples = [], []
for start in range(time_slices - window_size):
    segment = data[start:start + window_size, :]
    label = labels[start + window_size // 2]
    X_samples.append(segment)
    y_samples.append(label)

X_samples = np.array(X_samples)
y_samples = np.array(y_samples)

# Split into training/testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_samples, y_samples, test_size=0.2, random_state=42
)

# Build GRU model
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, input_shape=(window_size, num_features), return_sequences=True),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Plot results
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.title('Cyclone GRU Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Save model
model.save('models/cyclone_best_gru.h5')
