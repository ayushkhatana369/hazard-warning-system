from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf

app = Flask(__name__)
model = tf.keras.models.load_model('models/seismic_best_gru.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['spectrogram']  # expects shape [window_size, freq_bins]
        sample = np.array(data)
        # No channel dimension needed for GRU
        prediction = model.predict(np.array([sample])).flatten()[0]
        return jsonify({'probability': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
