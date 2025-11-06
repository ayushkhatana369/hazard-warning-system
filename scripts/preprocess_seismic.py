from obspy import read
import numpy as np
import matplotlib.pyplot as plt

# Load raw waveform
seismic_file = 'data/seismic/BW.BGLD..EH.D.2010.037'
stream = read(seismic_file)

# Select one trace for processing (example)
trace = stream[0]
data = trace.data

# Normalize the waveform data between -1 and 1
data_norm = data / np.max(np.abs(data))

# Plot normalized waveform segment
plt.figure(figsize=(10, 4))
plt.plot(data_norm[:2000])
plt.title('Normalized Seismic Waveform Segment')
plt.xlabel('Samples')
plt.ylabel('Amplitude (normalized)')
plt.grid(True)
plt.show()

# Save normalized data for ML model input (optional)
np.save('data/seismic/seismic_normalized.npy', data_norm)

