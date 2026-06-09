import pandas as pd
import requests
from io import StringIO

# Column names from the NOAA headers file
col_names = [
    'WBANNO','UTC_DATE','UTC_TIME','LST_DATE','LST_TIME','CRX_VN',
    'LONGITUDE','LATITUDE','T_CALC','T_HR_AVG','T_MAX','T_MIN',
    'P_CALC','SOLARAD','SOLARAD_FLAG','SOLARAD_MAX','SOLARAD_MAX_FLAG',
    'SOLARAD_MIN','SOLARAD_MIN_FLAG','SUR_TEMP_TYPE','SUR_TEMP',
    'SUR_TEMP_FLAG','SUR_TEMP_MAX','SUR_TEMP_MAX_FLAG','SUR_TEMP_MIN',
    'SUR_TEMP_MIN_FLAG','RH_HR_AVG','RH_HR_AVG_FLAG',
    'SOIL_MOISTURE_5','SOIL_MOISTURE_10','SOIL_MOISTURE_20',
    'SOIL_MOISTURE_50','SOIL_MOISTURE_100',
    'SOIL_TEMP_5','SOIL_TEMP_10','SOIL_TEMP_20','SOIL_TEMP_50','SOIL_TEMP_100'
]

# A Texas station with good soil sensor coverage across 2017
url = 'https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2026/CRNH0203-2026-CA_Santa_Barbara_11_W.txt'

raw = requests.get(url).text
df = pd.read_csv(StringIO(raw), sep=r'\s+', header=None, names=col_names)

# Replace missing value flags with NaN
df.replace(-9999.0, pd.NA, inplace=True)
df.replace(-99.000, pd.NA, inplace=True)

# Build datetime column
df['Time'] = pd.to_datetime(df['UTC_DATE'].astype(str) + df['UTC_TIME'].astype(str).str.zfill(4),
                             format='%Y%m%d%H%M', utc=True)

# Rename to match your dataset's structure
df_clean = df[[
    'Time',
    'RH_HR_AVG',       # → Humidity
    'T_HR_AVG',        # → Atmospheric_Temp
    'SOIL_TEMP_5',     # → Soil_Temp
    'SOIL_MOISTURE_5', # → Soil_Moisture (volumetric, 0-1 range)
    'P_CALC',          # → Rainfall (mm/hr) — the column you were missing!
]].rename(columns={
    'RH_HR_AVG':       'Humidity',
    'T_HR_AVG':        'Atmospheric_Temp',
    'SOIL_TEMP_5':     'Soil_Temp',
    'SOIL_MOISTURE_5': 'Soil_Moisture',
    'P_CALC':          'Rainfall',
})

# Compute Dew_Point from temp + humidity (Magnus formula)
import numpy as np
T = df_clean['Atmospheric_Temp']
RH = df_clean['Humidity']
df_clean['Dew_Point'] = T - ((100 - RH) / 5)

df_clean = df_clean.dropna(subset=['Soil_Moisture', 'Atmospheric_Temp'])
print(df_clean.shape)
print(df_clean.head())
print('\nAutocorr soil moisture lag1:', df_clean['Soil_Moisture'].autocorr(1))

df_clean.to_csv('uscrn_soil_2017_TX.csv', index=False)
print('Saved to uscrn_soil_2017_TX.csv')