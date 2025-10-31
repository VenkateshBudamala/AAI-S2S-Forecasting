# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:06:56 2023

@author: Dr. Venkatesh Budamala
"""

# Import necessary libraries
import os  # File path and directory operations
import pandas as pd  # Data handling and manipulation
import pygrib  # Reading GRIB files (weather data format)
import numpy as np  # Numerical operations
import time  # Measuring execution time
import xarray as xr  # Working with multidimensional data (NetCDF files)
from scipy.interpolate import griddata  # Interpolation for spatial data
import sqlite3  # Database management

# ---------------------------- Configuration ---------------------------- #
# Define model parameters
MODEL_NAME = 'NCEP'  # Name of the climate model
DATA_PATH = r'D:\Data\Climate_model_data\S2S'  # Path to model data
VAR_S2S = 'Minimum temperature at 2 metres in the last 6 hours'  # Variable name in GRIB files
SHORT_VAR = 'tmin'  # Short name for the variable
IMD_NC_FILE = r'D:\Data\Climate_observed_data\India\IMD\Temp\tmin_1951_2021.nc'  # IMD observed data file
OBSERVED_POINTS_FILE = r'D:\Results\S2S\India_grid_1deg_filter.csv'  # Observed grid points file
OUTPUT_PATH = r"C:\Users\User\Dropbox\Out_name"  # Path for output files
DB_NAME = 'S2S_India.db'  # SQLite database name

# Latitude and longitude range for India
LON_MIN, LON_MAX = 69.5, 97.5  # Longitude range
LAT_MIN, LAT_MAX = 8.5, 37.5  # Latitude range

# ---------------------------- Function Definitions ---------------------------- #
def extract_s2s_data(year, month):
    """Extracts S2S climate model data and interpolates it to a uniform grid."""
    s2s_array = np.empty((7, 29, 28))  # Placeholder for processed data
    
    path_s2s = os.path.join(DATA_PATH, 'S' + MODEL_NAME)  # Construct path to S2S model folder
    os.chdir(path_s2s)  # Change working directory to model data folder
    
    file_name = f"{year}_{month}_{MODEL_NAME}_PCP_TMAX_TMIN"  # Construct file name
    grbs = pygrib.open(file_name)  # Open GRIB file
    
    for step in range(7):  # Iterate over weekly steps (7 days)
        d2 = np.empty((28, 121, 240))  # Placeholder for daily temperature data
        grb_list = grbs.select(name=VAR_S2S)[step * 28:(step + 1) * 28]  # Select 28 daily records
        
        for i, grb in enumerate(grb_list):  # Loop through daily records
            d2[i, :, :] = grb.values  # Store values in array
        
        data = np.mean(d2, axis=0) - 273.15  # Convert Kelvin to Celsius and compute weekly mean
        lat, lon = grb_list[0].latlons()  # Extract latitude and longitude grids
        target_lon, target_lat = np.meshgrid(np.arange(LON_MIN, LON_MAX, 1), 
                                             np.arange(LAT_MIN, LAT_MAX, 1))  # Define target grid
        
        s2s_array[step, :, :] = griddata((lon.ravel(), lat.ravel()), data.ravel(),
                                          (target_lon, target_lat), method='cubic')  # Interpolate data
    
    grbs.close()  # Close GRIB file
    return s2s_array, target_lat, target_lon  # Return processed data and grid


def extract_imd_data(year, month):
    """Extracts IMD observed climate data for validation."""
    ds = xr.open_dataset(IMD_NC_FILE)  # Open NetCDF file
    start_date = f"{year}-{month:02d}-01"  # Define start date
    end_date = pd.to_datetime(start_date) + pd.DateOffset(weeks=6)  # Define end date (6 weeks later)
    ds_boundary = ds.sel(lon=slice(LON_MIN, LON_MAX),
                         lat=slice(LAT_MIN, LAT_MAX),
                         time=slice(start_date, end_date))  # Select data within bounds
    weekly_mean = ds_boundary.resample(time='7D').mean()  # Compute weekly mean
    return weekly_mean[SHORT_VAR].values  # Return processed data


def generate_parameter_sets(s2s_array, imd_obs, year, month, target_lat, target_lon):
    """Generates a structured dataset from model and observed data."""
    records = []  # Initialize list to store records
    
    for step in range(7):  # Iterate over weekly steps
        for row in range(29):  # Iterate over latitude grid
            for col in range(28):  # Iterate over longitude grid
                records.append([MODEL_NAME, year, month, step + 1, 
                                target_lat[row, col], target_lon[row, col], 
                                s2s_array[step, row, col], imd_obs[step, row, col]])  # Append data
    return records  # Return structured dataset


def save_results(data, filename):
    """Saves the processed data to CSV, SQLite, and HDF5."""
    df = pd.DataFrame(data, columns=['Model', 'Year', 'Month', 'Step', 'Latitude', 
                                     'Longitude', 'S2S', 'Obs'])  # Convert to DataFrame
    os.chdir(OUTPUT_PATH)  # Change directory to output path
    df.to_csv(filename + '.csv', index=False)  # Save as CSV file
    
    # Save to SQLite
    conn = sqlite3.connect(DB_NAME)  # Connect to SQLite database
    df.to_sql('Tmin', conn, index=False, if_exists='replace')  # Save table in database
    conn.close()  # Close database connection
    
    # Save to HDF5
    df.to_hdf(filename + '.h5', key='df', mode='w')  # Save as HDF5 file


# ---------------------------- Execution ---------------------------- #
if __name__ == "__main__":
    start_global_time = time.time()  # Record start time
    results = []  # Initialize results list
    
    for year in range(2015, 2022):  # Iterate over years
        for month in range(1, 13):  # Iterate over months
            print(f"Processing Year: {year}, Month: {month}")  # Print status
            start_time = time.time()  # Record start time for the month
            
            s2s_array, target_lat, target_lon = extract_s2s_data(year, month)  # Extract S2S data
            imd_obs = extract_imd_data(year, month)  # Extract IMD data
            records = generate_parameter_sets(s2s_array, imd_obs, year, month, target_lat, target_lon)  # Generate records
            
            results.extend(records)  # Append records to results
            end_time = time.time()  # Record end time
            print(f"Time taken for {year}-{month}: {end_time - start_time:.2f} seconds")  # Print execution time
            print("----------------------------------------------------------")  # Print separator
    
    save_results(results, SHORT_VAR + '_2015_2024_Nov_India_1')  # Save results
    end_global_time = time.time()  # Record global end time
    print(f"Total Execution Time: {end_global_time - start_global_time:.2f} seconds")  # Print total execution time
