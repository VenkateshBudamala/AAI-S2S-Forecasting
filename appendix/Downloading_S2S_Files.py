# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 17:06:56 2023

@author: Dr. Venkatesh Budamala
"""

from ecmwfapi import ECMWFDataServer
import os
import numpy as np

# Initialize ECMWFDataServer
server = ECMWFDataServer()

# Function to define the path for storing data
def set_data_path(path):
    os.chdir(path)

# Function to generate date range for a given year and month
def generate_date_range(year, month):
    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    days = days_in_month[month - 1]
    return f"{year}-{month:02d}-01/to/{year}-{month:02d}-{days}"

# Function to define the output filename format
def generate_filename(year, month, model_name):
    return f"{year}_{month:02d}_{model_name}_PCP_TMAX_TMIN.grib"

# Function to retrieve NCEP data from ECMWF server
def retrieve_data(year, month, date_range, output_filename):
    try:
        server.retrieve({
            "class": "s2",
            "dataset": "s2s",
            "date": date_range,
            "expver": "prod",
            "levtype": "sfc",
            "model": "glob",
            "origin": "kwbc",
            "param": "121/122/228228",  # Precipitation, Max Temp, Min Temp
            "step": "6/12/18/24/30/36/42/48/54/60/66/72/78/84/90/96/102/108/114/120/126/132/138/144/150/156/162/168/174/180/186/192/198/204/210/216/222/228/234/240/246/252/258/264/270/276/282/288/294/300/306/312/318/324/330/336/342/348/354/360/366/372/378/384/390/396/402/408/414/420/426/432/438/444/450/456/462/468/474/480/486/492/498/504/510/516/522/528/534/540/546/552/558/564/570/576/582/588/594/600/606/612/618/624/630/636/642/648/654/660/666/672/678/684/690/696/702/708/714/720/726/732/738/744/750/756/762/768/774/780/786/792/798/804/810/816/822/828/834/840/846/852/858/864/870/876/882/888/894/900/906/912/918/924/930/936/942/948/954/960/966/972/978/984/990/996/1002/1008/1014/1020/1026/1032/1038/1044/1050/1056",  # Steps for forecast data
            "stream": "enfo",
            "time": "00:00:00",
            "type": "cf",
            "target": output_filename
        })
        print(f"Data for {year}-{month:02d} saved as {output_filename}")
    except Exception as e:
        print(f"Error retrieving data for {year}-{month:02d}: {str(e)}")

# Function to automate the retrieval process for multiple years and months
def automate_data_retrieval(years, months, path):
    set_data_path(path)
    model_name = "NCEP"
    
    for year in years:
        for month in months:
            date_range = generate_date_range(year, month)
            output_filename = generate_filename(year, month, model_name)
            retrieve_data(year, month, date_range, output_filename)

# Define the years and months to retrieve
years = np.arange(2015, 2025)  # Example years 2015 to 2024
months = np.arange(1, 13)      # Months 1 to 12

# Define the path to store data
data_path = r"D:\Data"

# Run the automated data retrieval
automate_data_retrieval(years, months, data_path)
