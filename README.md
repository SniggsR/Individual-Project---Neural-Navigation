# Neural Navigation <br/>

<img src="https://user-images.githubusercontent.com/87940609/164897117-31678bb9-75fe-434b-a422-6215749b789b.jpg" width="600" height="350">

> Distance prediction without the need of GPS.
#### AIPI 540 NLP Individual Project | Spring Semester 2022

**Project By:** Snigdha Rudraraju

**Project:** Predicting Distance using accelerometer data and neural networks without the need of GPS.

## Getting Started

Download files in main.py folder
Make sure to place them at same location(same path)
Run streamlit.py
Upload accel.csv in processed data folder and click process.
Check the results tab

## Details about Equipment 

We used 3 devices here 

**Accelerometer**: Raspberry Pi Pico sensor is used in this project. It measures accelerations in X,Y,Z directions.

**GPS Sensor**: To get distance travelled using signals

**Microcontroller**: Collects the data in sequential order. Acceleration data was collected for every 200ms and GPS data was colleacted for every one second

## Data Preprocessing

Raw data is collected in a text file. It is converted into csv file and headers are added.Since the GPS data is collected for every one second where as acelerometer data is collected for every 200 milliseconds. Linear extrapolation is used to get the rest of the GPS data.

**Batching** 

Data is divided into batches of 40 seconds with stride of 1. We used a total of 675 batches.  

## Model

LSTM are used in this model. The network consists of 8 x 4 lstm nodes in a 2 layer network where accelerations in x,y,z directions are inputs and distance travelled is the output.

## Results

<img src="https://user-images.githubusercontent.com/87940609/165179398-c6498784-2708-4b60-bf37-05309086eeb7.png" width="600" height="350">









