from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS *after* creating the app instance

model = tf.keras.models.load_model('models/cyclone_best_gru.h5')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['spectrogram']  # expects [window_size, num_features]
        sample = np.array(data)
        prediction = model.predict(np.array([sample])).flatten()[0]
        return jsonify({'probability': float(prediction)})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
