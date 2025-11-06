import numpy as np
import tensorflow as tf
from tensorflow import keras
from kerastuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

# Load spectrogram data
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

X_train, X_test, y_train, y_test = train_test_split(X_samples, y_samples, test_size=0.2, random_state=42)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Conv2D(filters=hp.Int('filters_1', 16, 64, step=16),
                                  kernel_size=hp.Choice('kernel_size_1', [3, 5]),
                                  activation='relu',
                                  input_shape=X_train.shape[1:]))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    if hp.Boolean('second_conv'):
        model.add(keras.layers.Conv2D(filters=hp.Int('filters_2', 32, 128, step=32),
                                      kernel_size=hp.Choice('kernel_size_2', [3, 5]),
                                      activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu'))
    model.add(keras.layers.Dropout(rate=hp.Float('dropout', 0.0, 0.5, step=0.1)))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=keras.optimizers.Adam(
                      hp.Float('learning_rate', 1e-4, 1e-2, sampling='LOG')),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

tuner = RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    executions_per_trial=1,
    directory='keras_tuner_dir',
    project_name='seismic_cnn_tuning'
)

tuner.search(X_train, y_train,
             epochs=15,
             validation_data=(X_test, y_test),
             batch_size=32)

# Get the best model and save it
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
best_model.save('models/seismic_best_cnn.h5')
