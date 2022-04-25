# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 21:55:10 2022

@author: Kishan
"""
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from linearRegressionInference import linearRegressionInference as lRInf


class linearRegress(tf.keras.Model):
    def __init__(self, nInputs = 1, nOutputs = 1, structure=[10,10]):
        super(linearRegress, self).__init__(name="controller")
        self.nInputs = nInputs
        self.nOutputs = nOutputs
        self.structure = structure
        self.hidden_layers = []
        pass
    
    def build(self, input_shape):
        self.input_layer = tf.keras.layers.InputLayer(input_shape)
        for layer in range(len(self.structure)-1):
            self.hidden_layers.append(tf.keras.layers.Dense(self.structure[layer], activation=tf.tanh))
        self.output_layer = tf.keras.layers.Dense(self.nOutputs, activation=tf.tanh)
        pass
        
    def propagate(self, inputs):
        outputs = self.input_layer(inputs)
        for layer in range(len(self.structure)-1):
            outputs = self.hidden_layers[layer](outputs)
        outputs = self.output_layer(outputs)
        return outputs
    
    def call(self, inputs, training=True):
        return self.propagate(inputs)

class network:
    def __init__(self, nInputs=1, nOutputs=1, structure=[10,10]):
        self.plant = linearRegress(nInputs,nOutputs,structure)
        self.optimizer = tf.keras.optimizers.Adam(0.03)
        pass

    @tf.function
    def __call__(self, inputs, actualOutputs):
        with tf.GradientTape() as tape:
            outputs = self.plant(inputs)
            cost = tf.reduce_mean(tf.square(tf.subtract(actualOutputs,outputs)))
        gradients = tape.gradient(cost, self.plant.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.plant.trainable_variables))
        return outputs, cost
    
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
    inputs = [data['Ax-1'], data['Ay-1'], data['Az-1'], data['Ax'], data['Ay'], data['Az'], data['Ax-avg'], data['Ay-avg'], data['Az-avg']]
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
     
    inputsNormTest = inputsNorm0[:100,:]
    outputsNormTest = outputsNorm0[:100,:]
    inputsNorm = np.concatenate((inputsNorm0[100:,:],inputsNorm1,inputsNorm2),axis=0)
    outputsNorm = np.concatenate((outputsNorm0[100:,:],outputsNorm1,outputsNorm2),axis=0)
    
    allDataNorm = np.concatenate((inputsNorm, outputsNorm),axis=-1)
    np.random.shuffle(allDataNorm)
    
    #%%
    inputsNormTrain = allDataNorm[:,:-1]
    outputsNormTrain = allDataNorm[:,-1:]
    # inputsNormTest = allDataNorm[-50:,:,:3]
    # outputsNormTest = allDataNorm[-50:,:,3:]
    
    batchStartIndex = 0
    miniBatchSize = 100
    nBatches = miniBatchSize #inputsNormTrain.shape[0]
    structure = [1]
    net = network(inputsNorm.shape[-1],1,structure)
    time = np.linspace(0,timeSteps-1,timeSteps)
    testTime = np.linspace(0,outputsNormTest.shape[0]-1,outputsNormTest.shape[0])
#%%
    for i in range(1, 100000+1):
        if batchStartIndex >= inputsNormTrain.shape[0]-miniBatchSize:
            batchStartIndex = 0
            allDataNormTrain = np.concatenate((inputsNormTrain,outputsNormTrain),axis=-1) 
            np.random.shuffle(allDataNormTrain)
            inputsNormTrain = allDataNormTrain[:,:-1]
            outputsNormTrain = allDataNormTrain[:,-1:]
            # print('Data shuffled')
        miniBatchedInputs, miniBatchedOutputs = dataMiniBatcher(inputsNormTrain,outputsNormTrain,miniBatchSize,batchStartIndex)
        predOutputs, cost = net(miniBatchedInputs,miniBatchedOutputs)
        batchStartIndex += miniBatchSize
        
        if i%1000==0:
            weights = []
            biases = []
            for layer in range(len(net.plant.variables)//2):
                weights.append(np.asarray(net.plant.variables[2*layer]))
                biases.append(np.asarray(net.plant.variables[2*layer+1]))
            lRNet = lRInf(weights,biases,structure)
            predOutputsTest = lRNet.infer(inputsNormTest)
            predOutputsDenorm = (predOutputsTest+1)*(outputsMax-outputsMin)/2 + outputsMin
            testCost = np.mean(np.square(outputsNormTest-predOutputsTest))
            print('Iteration: {}, batchStartIndex: {}, Cost: {}, testCost: {}'.format(i,batchStartIndex,cost,testCost))
            
            plt.figure(1)
            plt.gcf().clear()
            plt.scatter(miniBatchedInputs[:,0],miniBatchedInputs[:,1])
            plt.scatter(miniBatchedInputs[:,2],miniBatchedInputs[:,3])
            plt.scatter(miniBatchedInputs[:,4],miniBatchedInputs[:,5])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)

            plt.figure(2)
            plt.gcf().clear()
            plt.scatter(miniBatchedOutputs[:,0],predOutputs[:,0])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)
            
            plt.figure(3)
            plt.gcf().clear()
            plt.scatter(inputsNormTest[:,0],inputsNormTest[:,1])
            plt.scatter(inputsNormTest[:,2],inputsNormTest[:,3])
            plt.scatter(inputsNormTest[:,4],inputsNormTest[:,5])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)

            plt.figure(4)
            plt.gcf().clear()
            plt.scatter(outputsNormTest[:,0],predOutputsTest[:,0])
            plt.ylim(top=1,bottom=-1)
            plt.pause(0.1)
            
            
#%%
# import scipy.io as sio
# wnb = {}
# layerNum = 0
# i = 0
# while i < (len(structure)+1)*2:
#     wnb['w'+str(layerNum)] = np.asarray(net.plant.variables[i])
#     wnb['b'+str(layerNum)] = np.asarray(net.plant.variables[i+1])
#     layerNum+=1
#     i+=2
# wnb['inputsMax'] = inputsMax
# wnb['inputsMin'] = inputsMin
# wnb['outputsMax'] = outputsMax
# wnb['outputsMin'] = outputsMin
# sio.savemat('wnb.mat',wnb)

#%%
