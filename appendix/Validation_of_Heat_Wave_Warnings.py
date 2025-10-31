# -*- coding: utf-8 -*-
"""
Created on Tue Apr  2 15:50:42 2024

@author: Dr. Venkatesh Budamala
"""
from IPython import get_ipython
get_ipython().magic('reset -sf')

import os
import pandas as pd
import numpy as np
import xarray as xr
import sqlite3

##### IMD data ####
os.chdir('D:\\Data\\Climate_observed_data\\India\\IMD\\Temp')
data = xr.open_dataset('tmax_1951_2021.nc')
data = data.sel(time = slice(None, '2014'))
monthly_mean_all_years = data.groupby('time.month').mean(dim='time')
monthly_df = monthly_mean_all_years.to_dataframe()


##### Model Data #####
os.chdir(r'D:\Data\Climate_model_data\S2S\Save\UPD')
conn = sqlite3.connect('S2S_India.db')
vname1 = "SELECT * FROM Tmax_ML"
df1 = pd.read_sql_query(vname1, conn)
conn.close()


df4 = pd.DataFrame()
for mon in range(3,7):

    for lead in range(1,7):
        
        df_mon = df1[df1['Month'] == mon]
        df_mon_lead = df_mon[df_mon['Step'] == lead]
        
        
        for iter1 in range(len(df_mon_lead.iloc[:,0])):
            
            df2 = df_mon_lead.iloc[[iter1]]
            
            df2.index = np.arange(1, len(df2) + 1)
            
            if lead >= 5:
               selected_month = df2['Month'].iloc[0]+1
               
            selected_month = df2['Month'].iloc[0]
            selected_lat = df2['Latitude'].iloc[0]
            selected_lon = df2['Longitude'].iloc[0]
            
               
            val = monthly_df.loc[(selected_month, selected_lat, selected_lon), 'tmax']
            
            val1 = val + 4.5 # HW
            val2 = val + 6.4 # SHW
            
            df3 = df2.iloc[:,-3:]
            result1 = ((df3 > val1) & (df3 <val2)).astype(int)
            result1.columns = ['S2S_HW', 'ML_HW', 'Obs_HW']
            
            df2 = pd.concat([df2, result1], axis=1)
            
            result2 = (df3 > val2).astype(int)
            result2.columns = ['S2S_SHW', 'ML_SHW', 'Obs_SHW']
            df2 = pd.concat([df2, result2], axis=1)
            
            df4 = pd.concat([df4,df2], axis = 0)
            df4.index = np.arange(1, len(df4) + 1)


conn = sqlite3.connect('S2S_India.db')
vname2 = "Tmax_ML_HW_SHW_UPD1"

df4.to_sql(vname2, conn, index=False, if_exists='replace')
conn.close()

