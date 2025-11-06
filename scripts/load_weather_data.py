import netCDF4
import matplotlib.pyplot as plt

# Path to your NetCDF file
nc_file = 'data/cyclone/datasettt.nc'

# Load the dataset
dataset = netCDF4.Dataset(nc_file)

# Print available variables (you can comment this out later)
print(dataset.variables.keys())

# Extract cyclone track Latitude and Longitude variables
lats = dataset.variables['lat'][:]
lons = dataset.variables['lon'][:]

# Plot cyclone track coordinates
plt.figure(figsize=(10, 6))
plt.plot(lons, lats, 'b.', markersize=2)
plt.title('Cyclone Track Coordinates')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(True)
plt.show()
