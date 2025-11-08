from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import traceback

app = Flask(__name__)
CORS(app)

model_path = 'models/seismic_best_gru.h5'

try:
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Loaded model: {model_path}")
except Exception as e:
    print(f"❌ Failed to load model from {model_path}: {e}")
    model = None


@app.route('/predict', methods=['POST'])
def predict():
    try:
        print("\n=== 🛰️ Incoming Request ===")
        print("Raw JSON received:", request.json)

        if model is None:
            return jsonify({'error': 'Model not loaded on server.'}), 500

        data = request.json.get('spectrogram')
        if data is None:
            return jsonify({'error': "Missing 'spectrogram' key in request"}), 400

        sample = np.array(data)
        print("Original sample shape:", sample.shape)

        # GRU expects 3D input: (batch_size, timesteps, features)
        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=0)
        elif sample.ndim == 1:
            sample = np.expand_dims(np.expand_dims(sample, axis=0), axis=-1)

        print("Prepared input shape:", sample.shape)

        prediction = model.predict(sample).flatten()[0]
        print("Model output probability:", prediction)

        return jsonify({'probability': float(prediction)})

    except Exception as e:
        print("❌ Full error trace:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
