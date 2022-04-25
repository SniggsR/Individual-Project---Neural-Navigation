# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 20:20:08 2022

@author: Kishan

Source: 
    1. https://www.geeksforgeeks.org/program-distance-two-points-earth/#:~:text=For%20this%20divide%20the%20values,is%20the%20radius%20of%20Earth.
    2. https://stackoverflow.com/questions/45794490/replace-zeros-in-numpy-array-with-linear-interpolation-between-its-preceding-and
    3. https://stackoverflow.com/questions/45429831/valueerror-a-value-in-x-new-is-above-the-interpolation-range-what-other-re

"""

import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

def distance(lat, lon): #Returns distance between two latitude and longitudes in meters
    lat = np.radians(lat)
    lon = np.radians(lon)
    dLat = lat[1:]-lat[:-1]
    dLon = lon[1:]-lon[:-1]
    distance = 6371000*2*np.arcsin(np.sqrt(np.sin(dLat/2)**2+np.cos(lat[:-1])*np.cos(lat[1:])*np.sin(dLon/2)**2))
    distance = np.append([0.0],distance)
    nonZeroDistanceIndices = np.nonzero(distance)
    cumulativeDistance = np.zeros([len(distance)])
    cumulativeDistance[nonZeroDistanceIndices] = np.cumsum(distance[nonZeroDistanceIndices])
    interp = interp1d(np.arange(len(cumulativeDistance))[nonZeroDistanceIndices],cumulativeDistance[nonZeroDistanceIndices],fill_value='extrapolate')
    cumulativeDistanceInterpolated = interp(np.arange(len(cumulativeDistance)))
    return distance, cumulativeDistance, cumulativeDistanceInterpolated


fileLoc = './data/Norwalk/gps_run2.csv'

rawData = pd.read_csv(fileLoc)

if 'Distance' in rawData:
    print('Distance already exists in the given file')
else: 
    print('Calculating Distance')
    lat = np.asarray(rawData['Latitude'])
    lon = np.asarray(rawData['Longitude'])
    
    
    distance, cumulativeDistance, cumulativeDistanceInterpolated = distance(lat,lon)
    
    rawData['Distance'] = cumulativeDistanceInterpolated
    try: rawData.to_csv(fileLoc, index=False); print('Distance has been calculated and updated in the csv file')
    except: print('Unable to open/write to the file. Please close it and rerun the code')

