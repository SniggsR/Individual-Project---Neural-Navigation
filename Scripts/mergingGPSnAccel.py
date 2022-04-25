# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:40:12 2022

@author: Kishan
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d

fileLoc = './data/Norwalk/'

accelData = pd.read_csv(fileLoc+'accel_run2.csv')
gpsData = pd.read_csv(fileLoc+'gps_run2.csv')

interp = interp1d(np.asarray(gpsData['Tics']),np.asarray(gpsData['Distance']))
interpDistance = interp(np.asarray(accelData['Tics']))

if 'Distance' in accelData:
    print('Distance already exists in the given file')
else: 
    print('Calculating Distance')
    accelData['Distance'] = interpDistance
    try: accelData.to_csv(fileLoc+'accel_run2.csv', index=False); print('Distance has been calculated and updated in the csv file')
    except: print('Unable to open/write to the file. Please close it and rerun the code')
