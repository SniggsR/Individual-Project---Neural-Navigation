#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 19 13:11:57 2022

@author: snigdharudraraju
"""
import pandas as pd
import streamlit as st
from PIL import Image
import scipy.io as sio
from iLSTMInference import iLSTMInference as lstmInf
import numpy as np



def main():
    
    menu=["Home","Dataset","Results"]
    choice = st.sidebar.selectbox("Menu",menu) 
    
    if choice == "Home":
        st.title("Holaaa")
        st.subheader("What are we solving exactly?")
        st.markdown ("""In this project, We are using accelerometer(X,Y,Z axis) 
                     and GPS sensor data connected to Raspberry Pi Pico microsensor for predicting 
                     your current speed. We will use the accelerometer as input and GPS data as output to build a neural network.This neural network will then use the data given by users to predict the distance and speed travelled with out the use of GPS sensor.
                     We are trying to solve the problem of using GPS sensors which are both costly and do not work accurately in closed spaces.
                     
                     """)
        st.subheader("Go and check it out!!!")
        
    elif choice=="Dataset":
        st.title("Lets see how much you walked")
        st.subheader("Upload the data below")
    
    
        csv_file = st.file_uploader("Upload accelerometer CSV file and check your speed",
                                    type=["csv"])
        
        if csv_file is not None:
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
            weights = []
            biases = []
            for i in range(np.int(len(wnb)/2)):
                weights.append(wnb['w'+str(i)])
                biases.append(wnb['b'+str(i)])
            #to see details of the files
            st.write(type(csv_file))
            
            file_details = {"Filename": csv_file.name,
                            "Filetype":csv_file.type , "Filesize": csv_file.size}
        
            st.write(file_details)
            df = pd.read_csv(csv_file)
            st.dataframe(df)
            st.button("Process")
            
            inputs = [df['Ax'], df['Ay'], df['Az']]
            inputs = np.asarray(inputs,dtype=np.float32).transpose()
            outputs = np.asarray(df['DistanceDelta'],dtype=np.float32).reshape([inputs.shape[0],1])

            inputsNorm = -1+(2*(inputs-inputsMin)/(inputsMax-inputsMin))
            outputsNorm = -1+(2*(outputs-outputsMin)/(outputsMax-outputsMin))
            
            del wnb, i
            
            lstmNet = lstmInf(weights,biases)
            
            predOutputs = lstmNet.infer(inputsNorm)
            predOutputsDenorm = (predOutputs+1)*(outputsMax-outputsMin)/2 + outputsMin
            print(np.sum(predOutputsDenorm))
            
    elif choice=="Results":
        st.title("Time for results")
    

if __name__ == '__main__':
    main()