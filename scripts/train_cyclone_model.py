import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load preprocessed cyclone data
lats = np.load('data/cyclone/cyclone_lats.npy')
lons = np.load('data/cyclone/cyclone_lons.npy')
winds = np.load('data/cyclone/cyclone_winds.npy')

# For this demo, let's classify wind speed into two classes (e.g., strong vs weak)
threshold = 50  # knots
y = (winds > threshold).astype(int)

# Features: lat and lon coordinates
X = np.vstack((lats, lons)).T

# Train-test split 80-20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build simple logistic regression-like NN model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Cyclone Wind Speed Classification Accuracy')
plt.show()

# Save model
model.save('models/cyclone_wind_classifier.h5')

