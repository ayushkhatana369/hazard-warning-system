from obspy import read
import matplotlib.pyplot as plt

# Path to your seismic data file
seismic_file = 'data/seismic/BW.BGLD..EH.D.2010.037'


# Read and print the waveform stream
stream = read(seismic_file)
print(stream)

# Plot the waveform
stream.plot()
plt.show()
