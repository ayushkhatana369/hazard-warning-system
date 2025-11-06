import numpy as np
from obspy import read
from scipy.signal import spectrogram
import matplotlib.pyplot as plt

# Load seismic waveform
seismic_file = 'data/seismic/BW.BGLD..EH.D.2010.037'
stream = read(seismic_file)
trace = stream[0]
data = trace.data

# Normalize amplitude
data_norm = data / np.max(np.abs(data))

# Define segment length and overlap for spectrogram
segment_length = 256
overlap = 128

# Calculate spectrogram (frequencies, times, spectrogram matrix)
frequencies, times, Sxx = spectrogram(data_norm, nperseg=segment_length, noverlap=overlap)

# Convert to log scale for better visualization and modeling
Sxx_log = np.log(Sxx + 1e-10)

# Plot example spectrogram
plt.pcolormesh(times, frequencies, Sxx_log, shading='gouraud')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Seismic Signal')
plt.colorbar(label='Log Power')
plt.show()

# Save spectrogram for ML use
np.save('data/seismic/spectrogram.npy', Sxx_log)
