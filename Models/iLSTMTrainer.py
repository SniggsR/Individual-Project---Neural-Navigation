# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:55:10 2022

@author: Kishan
"""
import tensorflow as tf
from iLSTM import core as iLSTMCore
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from iLSTMInference import iLSTMInference as lstmInf


class lstmModel(tf.keras.Model):
    def __init__(self, nInputs = 1, nOutputs = 1, structure=[4, 2]):
        super(lstmModel, self).__init__(name="controller")
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.structure = structure
        pass
    
    def build(self, input_shape):
        self.input_layer = tf.keras.layers.InputLayer(input_shape)
        self.core = iLSTMCore(self.structure)
        self.output_layer = tf.keras.layers.Dense(self.nOutputs, activation=tf.tanh)
        pass
    
    def createMemory(self, input_shape):
        shortTermMemory, longTermMemory = self.core.createMemory(input_shape)
        return shortTermMemory, longTermMemory
    
    def propagate(self, inputs, shortTermMemory, longTermMemory):
        outputs = self.input_layer(inputs)
        outputs, shortTermMemory, longTermMemory = self.core(outputs, shortTermMemory, longTermMemory)
        outputs = self.output_layer(outputs)*1
        return outputs, shortTermMemory, longTermMemory
    
    def call(self, inputs, shortTermMemory, longTermMemory, training=True):
        return self.propagate(inputs, shortTermMemory, longTermMemory)

class network:
    def __init__(self, nInputs=1, nOutputs=1, structure=[4,2]):
        self.plant = lstmModel(nInputs,nOutputs,structure)
        self.optimizer = tf.keras.optimizers.Adam(0.03)
        pass
    
    @tf.function
    def __call__(self, inputs, actualOutputs, shortTermMemory, longTermMemory):
        outputs = []
        with tf.GradientTape() as tape:
            for i in range(inputs.shape[1]):
                # if i%100==0:
                #     print(i)
                output, shortTermMemory, longTermMemory = self.plant(inputs[:,i:i+1,:], shortTermMemory, longTermMemory)
                outputs.append(output)
            outputs = tf.concat(outputs, axis=-2)
            cost = tf.reduce_mean(tf.square(tf.subtract(actualOutputs,outputs))) + 0.001*(tf.reduce_mean(tf.square(net.plant.variables[0]))
                                                                                          +tf.reduce_mean(tf.square(net.plant.variables[1]))
                                                                                          +tf.reduce_mean(tf.square(net.plant.variables[2]))
                                                                                          +tf.reduce_mean(tf.square(net.plant.variables[3]))
                                                                                          +tf.reduce_mean(tf.square(net.plant.variables[4]))
                                                                                          +tf.reduce_mean(tf.square(net.plant.variables[5])))
        gradients = tape.gradient(cost, self.plant.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.plant.trainable_variables))
        return outputs, cost
    
def dataBatcher(inputs,outputs,batchSize,stride=1):
    batchedInputs = []
    batchedOutputs = []
    for i in range(inputs.shape[0]-batchSize):
        batchedInputs.append(inputs[i*stride:i*stride+batchSize,...])
        batchedOutputs.append(outputs[i*stride:i*stride+batchSize,...])
    batchedInputs = np.stack(batchedInputs)
    batchedOutputs = np.stack(batchedOutputs)
    return batchedInputs, batchedOutputs

def dataAugment(data,batchSize,stride=1):
    augmentedData = []
    for i in range((data.shape[0]-batchSize)//stride+1):
        augmentedData.append(data[np.newaxis,i*stride:i*stride+batchSize,:])
    return np.concatenate(augmentedData)

def dataMiniBatcher(batchedInputs,batchedOutputs,miniBatchSize,batchStartIndex):
    miniBatchedInputs = batchedInputs[batchStartIndex:batchStartIndex+miniBatchSize,...]
    miniBatchedOutputs = batchedOutputs[batchStartIndex:batchStartIndex+miniBatchSize,...]
    return miniBatchedInputs, miniBatchedOutputs

def testPrediction(plant, inputsNormTest, shortTermMemoryTest, longTermMemoryTest):
    outputs = []
    for i in range(inputsNormTest.shape[1]):
        output,shortTermMemoryTest,longTermMemoryTest = plant(inputsNormTest[:,i:i+1,:],shortTermMemoryTest,longTermMemoryTest)
        outputs.append(output)
    outputs = tf.concat(outputs, axis=-2)
    return outputs

def dataImport(data):
    inputs = [data['Ax'], data['Ay'], data['Az']]
    inputs = np.asarray(inputs,dtype=np.float32).transpose()
    outputs = np.asarray(data['DistanceDelta'],dtype=np.float32).reshape([inputs.shape[0],1])
    return inputs, outputs

#%%
if __name__ == "__main__":
    fileLoc = './data/'
    data0 = pd.read_csv(fileLoc+'accel.csv')
    data1 = pd.read_csv(fileLoc+'Norwalk/accel_run1.csv')
    data2 = pd.read_csv(fileLoc+'Norwalk/accel_run2.csv')
    
    inputs0,outputs0 = dataImport(data0)
    inputs1,outputs1 = dataImport(data1)
    inputs2,outputs2 = dataImport(data2)

    inputs0 = inputs0[10:-10,...]
    outputs0 = outputs0[10:-10,...]
    inputs1 = inputs1[500:501,...]
    outputs1 = outputs1[500:501,...]
    inputs2 = inputs2[500:501,...]
    outputs2 = outputs2[500:501,...]

    inputsMax = np.max(np.concatenate((inputs0,inputs1,inputs2),axis=0),axis=0)
    inputsMin = np.min(np.concatenate((inputs0,inputs1,inputs2),axis=0),axis=0)

    outputsMax = np.max(np.concatenate((outputs0,outputs1,outputs2),axis=0),axis=0)
    outputsMin = np.min(np.concatenate((outputs0,outputs1,outputs2),axis=0),axis=0)
    
    inputsNorm0 = -1+(2*(inputs0-inputsMin)/(inputsMax-inputsMin))
    inputsNorm1 = -1+(2*(inputs1-inputsMin)/(inputsMax-inputsMin))
    inputsNorm2 = -1+(2*(inputs2-inputsMin)/(inputsMax-inputsMin))
    outputsNorm0 = -1+(2*(outputs0-outputsMin)/(outputsMax-outputsMin))
    outputsNorm1 = -1+(2*(outputs1-outputsMin)/(outputsMax-outputsMin))
    outputsNorm2 = -1+(2*(outputs2-outputsMin)/(outputsMax-outputsMin))
    
    allDataNorm0 = np.concatenate((inputsNorm0,outputsNorm0),axis=-1)
    allDataNorm1 = np.concatenate((inputsNorm1,outputsNorm1),axis=-1)
    allDataNorm2 = np.concatenate((inputsNorm2,outputsNorm2),axis=-1)
    
    timeSteps = 200
 
    batchedAllDataNorm0 = dataAugment(allDataNorm0,timeSteps,stride=1)
    # batchedAllDataNorm1 = dataAugment(allDataNorm1,timeSteps,stride=7)
    # batchedAllDataNorm2 = dataAugment(allDataNorm2,timeSteps,stride=7)
    
    inputsNormTest = inputsNorm0
    outputsNormTest = outputsNorm0
    inputsNorm = batchedAllDataNorm0[...,:-1] #np.concatenate((batchedAllDataNorm1[...,:-1],batchedAllDataNorm2[...,:-1]),axis=0)
    outputsNorm = batchedAllDataNorm0[...,-1:] #np.concatenate((batchedAllDataNorm1[...,-1:],batchedAllDataNorm2[...,-1:]),axis=0)
    
    allDataNorm = np.concatenate((inputsNorm, outputsNorm),axis=-1)
    np.random.shuffle(allDataNorm)
    
    #%%
    inputsNormTrain = allDataNorm[:,:,:-1]
    outputsNormTrain = allDataNorm[:,:,-1:]
    # inputsNormTest = allDataNorm[-50:,:,:3]
    # outputsNormTest = allDataNorm[-50:,:,3:]
    
    batchStartIndex = 0
    miniBatchSize = 100
    nBatches = miniBatchSize #inputsNormTrain.shape[0]
    structure = [8,4]
    net = network(3,1,structure)
    shortTermMemory = tf.Variable(tf.zeros([nBatches, 1, sum(structure)], dtype=tf.float32), trainable=True, name='shortTermMemory')
    longTermMemory = tf.Variable(tf.zeros([nBatches, 1, sum(structure)], dtype=tf.float32), trainable=True, name='longTermMemory')
    shortTermMemoryTest = tf.Variable(tf.zeros([nBatches, 1, sum(structure)], dtype=tf.float32), trainable=True, name='shortTermMemory')
    longTermMemoryTest = tf.Variable(tf.zeros([nBatches, 1, sum(structure)], dtype=tf.float32), trainable=True, name='longTermMemory')
    time = np.linspace(0,timeSteps-1,timeSteps)
    testTime = np.linspace(0,outputsNormTest.shape[0]-1,outputsNormTest.shape[0])
#%%
    for i in range(1, 10000+1):
        if batchStartIndex >= inputsNormTrain.shape[0]-miniBatchSize:
            batchStartIndex = 0
            allDataNormTrain = np.concatenate((inputsNormTrain,outputsNormTrain),axis=-1) 
            np.random.shuffle(allDataNormTrain)
            inputsNormTrain = allDataNormTrain[:,:,:-1]
            outputsNormTrain = allDataNormTrain[:,:,-1:]
            # print('Data shuffled')
        miniBatchedInputs, miniBatchedOutputs = dataMiniBatcher(inputsNormTrain,outputsNormTrain,miniBatchSize,batchStartIndex)
        predOutputs, cost = net(miniBatchedInputs,miniBatchedOutputs,shortTermMemory,longTermMemory)
        batchStartIndex += miniBatchSize
        
        if i%100==0:
            weights = []
            biases = []
            layerNum = 0
            j = 0
            while j < (len(structure)+1)*2:
                weights.append(np.asarray(net.plant.variables[j]))
                biases.append(np.asarray(net.plant.variables[j+1]))
                layerNum+=1
                j+=2
            lstmNet = lstmInf(weights,biases)
            predOutputsTest = lstmNet.infer(inputsNormTest)
            predOutputsDenorm = (predOutputsTest+1)*(outputsMax-outputsMin)/2 + outputsMin
            testCost = np.mean(np.square(outputsNormTest-predOutputsTest))
            batchToPlot = np.random.randint(0,9)
            print('Iteration: {}, batchStartIndex: {}, Cost: {}, testCost: {}'.format(i,batchStartIndex,cost,testCost))
            
            plt.figure(1)
            plt.gcf().clear()
            plt.plot(time,miniBatchedInputs[batchToPlot,:,0])
            # plt.plot(time,miniBatchedInputs[batchToPlot,:,1])
            # plt.plot(time,miniBatchedInputs[batchToPlot,:,2])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)

            plt.figure(2)
            plt.gcf().clear()
            plt.plot(time,miniBatchedOutputs[batchToPlot,:,0])
            plt.plot(time,predOutputs[batchToPlot,:,0])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)
            
            plt.figure(3)
            plt.gcf().clear()
            plt.plot(testTime,inputsNormTest[:,0])
            # plt.plot(testTime,inputsNormTest[:,1])
            # plt.plot(testTime,inputsNormTest[:,2])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)

            plt.figure(4)
            plt.gcf().clear()
            plt.plot(testTime,outputsNormTest[:,0])
            plt.plot(testTime,predOutputsTest[:,0])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)
            
            
#%%
import scipy.io as sio
wnb = {}
layerNum = 0
i = 0
while i < (len(structure)+1)*2:
    wnb['w'+str(layerNum)] = np.asarray(net.plant.variables[i])
    wnb['b'+str(layerNum)] = np.asarray(net.plant.variables[i+1])
    layerNum+=1
    i+=2
wnb['inputsMax'] = inputsMax
wnb['inputsMin'] = inputsMin
wnb['outputsMax'] = outputsMax
wnb['outputsMin'] = outputsMin
sio.savemat('wnb.mat',wnb)

#%%
