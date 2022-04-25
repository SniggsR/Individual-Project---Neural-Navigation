# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 15:42:24 2022

@author: Kishan
"""

import scipy.io as sio
import pandas as pd
from iLSTMInference import iLSTMInference as lstmInf
import numpy as np

#Importing accelerometer data
fileLoc = './data/'
data = pd.read_csv(fileLoc+'accel.csv')
inputs = [data['Ax'], data['Ay'], data['Az']]
inputs = np.asarray(inputs,dtype=np.float32).transpose()
outputs = np.asarray(data['DistanceDelta'],dtype=np.float32).reshape([inputs.shape[0],1])

#Importing weights and biases of the trained model
wnb = sio.loadmat('wnb.mat')
inputsMax = wnb['inputsMax']
inputsMin = wnb['inputsMin']
outputsMax = wnb['outputsMax']
outputsMin = wnb['outputsMin']
wnb.pop('__globals__')
wnb.pop('__header__')
wnb.pop('__version__')
wnb.pop('inputsMax')
wnb.pop('inputsMin')
wnb.pop('outputsMax')
wnb.pop('outputsMin')

inputsNorm = -1+(2*(inputs-inputsMin)/(inputsMax-inputsMin))
outputsNorm = -1+(2*(outputs-outputsMin)/(outputsMax-outputsMin))

weights = []
biases = []
for i in range(np.int(len(wnb)/2)):
    weights.append(wnb['w'+str(i)])
    biases.append(wnb['b'+str(i)])

del wnb, i, data

lstmNet = lstmInf(weights,biases)

predOutputs = lstmNet.infer(inputsNorm)
predOutputsDenorm = (predOutputs+1)*(outputsMax-outputsMin)/2 + outputsMin