import requests
import numpy as np

window_size = 64
num_features = 6

sample = np.random.rand(window_size, num_features).tolist()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    json={"spectrogram": sample}
)

print('Status code:', response.status_code)
print('Response:', response.json())
