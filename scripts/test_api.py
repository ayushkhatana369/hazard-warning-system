import requests
import numpy as np

frequency_bins = 129  # Set this to your spectrogram freq dimension used in training
window_size = 64      # Same window size used for training

# Create a random spectrogram sample with correct shape for GRU
sample = np.random.rand(window_size, frequency_bins).tolist()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"spectrogram": sample}
)

print('Status code:', response.status_code)
print('Raw response:', response.text)
try:
    print('JSON:', response.json())
except Exception as e:
    print('Failed to parse JSON:', e)
