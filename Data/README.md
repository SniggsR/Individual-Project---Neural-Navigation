
#### Data Collection and Pre Processing

## Collection

The data is collected seperately from accelerometer(for every 200 milliseconds) and GPS sensors(For every one second) with use of micro controller in a timely manner. The code for getting data in written in micropython for the microcontroller to work.

**Raw** 

This folder containes files which are collected from microcontroller. Initially they were txt files without any headers. basic cleaning was done.

**Processed**

This folder contains files processed files which are used as final inputs for the model. The input thats needs to be used for running the model is accel.csv.

**Scripts used to get the final data will be explained in Scripts folder**

## How are Inputs Batched

We have used concept of timesteps here. Lets discuss how many timesteps were used 

Total number of timesteps = 875

Each timestep = 200 milliseconds


Accelerations in X direction = 875 ; Accelerations in Y direction = 875 ; Accelerations in Z direction = 875

Initial Inputs = 875 timesteps x 3 directions {875 * 3}

Initial Outputs = 875 distances x  1 {875 * 1}

Shape of mini batch == 200 time steps == 200 * 200 milliseconds(time of each timestep) = 40 seconds

Input after converting to batches = 675 * 200 * 3

Output after converting to batches = 675 * 200 * 1





