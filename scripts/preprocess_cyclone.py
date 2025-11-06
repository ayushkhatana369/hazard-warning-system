import netCDF4
import numpy as np
import matplotlib.pyplot as plt

# Load IBTrACS NetCDF file
nc_file = 'data/cyclone/datasettt.nc'
ds = netCDF4.Dataset(nc_file)

# Extract variables for one storm (first in set for simplicity)
lats = ds.variables['lat'][:]   # shape: [obs,]
lons = ds.variables['lon'][:]
winds = ds.variables['wmo_wind'][:]

# Filter out invalid values (e.g., -999.0 missings)
valid_idx = (lats < 99) & (lons < 999) & (winds > 0)  # Adjust as necessary
lats, lons, winds = lats[valid_idx], lons[valid_idx], winds[valid_idx]

# Plot cyclone tracks (all valid points)
plt.figure(figsize=(10, 6))
plt.scatter(lons, lats, c=winds, cmap='viridis', s=10)
plt.colorbar(label='Wind Speed (knots)')
plt.title('Cyclone Observations (Wind Colored)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()

# Save preprocessed arrays for ML
lats_filled = lats.astype(float).filled(np.nan)
lons_filled = lons.astype(float).filled(np.nan)
winds_filled = winds.astype(float).filled(np.nan)

np.save('data/cyclone/cyclone_lats.npy', lats_filled)
np.save('data/cyclone/cyclone_lons.npy', lons_filled)
np.save('data/cyclone/cyclone_winds.npy', winds_filled)



