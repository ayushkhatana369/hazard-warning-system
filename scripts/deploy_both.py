from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import traceback

app = Flask(__name__)
CORS(app)   

# Load Cyclone model
try:
    cyclone_model = tf.keras.models.load_model('models/cyclone_best_gru.h5')
    print("✅ Loaded cyclone model")
except Exception as e:
    print(f"❌ Failed to load cyclone model: {e}")
    cyclone_model = None

# Load Earthquake (Seismic) model
try:
    earthquake_model = tf.keras.models.load_model('models/seismic_best_gru.h5')
    print("✅ Loaded earthquake model")
except Exception as e:
    print(f"❌ Failed to load earthquake model: {e}")
    earthquake_model = None

def validate_input(data, expected_rows=64, expected_cols=None):
    if not isinstance(data, list):
        return False
    if len(data) not in [1, expected_rows]:  # Allow single row or full expected rows
        return False
    for row in data:
        if not isinstance(row, list):
            return False
        if len(row) != expected_cols:
            return False
    return True

@app.route('/predict/cyclone', methods=['POST'])
def predict_cyclone():
    if cyclone_model is None:
        return jsonify({'error': 'Cyclone model not loaded'}), 500
    try:
        data = request.json.get('spectrogram')
        if data is None:
            return jsonify({'error': "Missing 'spectrogram' key"}), 400

        # Validate input for cyclone: 64 (or 1) rows and 6 columns
        if not validate_input(data, expected_rows=64, expected_cols=6):
            return jsonify({'error': "Input must be 64 rows (or 1 row) with 6 columns"}), 400

        sample = np.array(data)
        sample = np.expand_dims(sample, axis=0)  # Add batch dimension

        prediction = cyclone_model.predict(sample).flatten()[0]
        return jsonify({'probability': float(prediction)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

@app.route('/predict/earthquake', methods=['POST'])
def predict_earthquake():
    if earthquake_model is None:
        return jsonify({'error': 'Earthquake model not loaded'}), 500
    try:
        data = request.json.get('spectrogram')
        if data is None:
            return jsonify({'error': "Missing 'spectrogram' key"}), 400

        # Validate input for earthquake: 64 (or 1) rows and 129 columns
        if not validate_input(data, expected_rows=64, expected_cols=129):
            return jsonify({'error': "Input must be 64 rows (or 1 row) with 129 columns"}), 400

        sample = np.array(data)
        # Model expects 3D input: (batch, timesteps, features)
        if sample.ndim == 2:
            sample = np.expand_dims(sample, axis=0)
        elif sample.ndim == 1:
            sample = np.expand_dims(np.expand_dims(sample, axis=0), axis=-1)

        prediction = earthquake_model.predict(sample).flatten()[0]
        return jsonify({'probability': float(prediction)})

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
