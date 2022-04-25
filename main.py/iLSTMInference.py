# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:33:04 2022

@author: Kishan
"""

import numpy as np

class iLSTMInference:
    def __init__(self, weights, biases, initSTM=None, initLTM=None):
        self.weights = weights
        self.biases = biases
        self.numOfLayers = len(self.weights)-1 
        self.structure = []
        self.outputs = []
        for i in range(self.numOfLayers):
            self.structure.append(np.int(self.biases[i].shape[1]/4))
        self.numOfInputs = np.int(self.weights[0].shape[0]-self.structure[0])
        self.numOfOutputs = np.int(self.biases[-1].shape[-1])
        if initSTM==None:
            self.stm = []
            self.ltm = []
            for i in range(self.numOfLayers):
                self.stm.append(np.zeros([1,self.structure[i]]))
                self.ltm.append(np.zeros([1,self.structure[i]]))
        else:
            self.stm = initSTM
            self.ltm = initLTM
        pass
    
    def sigmoid(self,inputs):
        return 1/(1+np.exp(-inputs))
    
    def infer(self,inputs):
        for timestep in range(inputs.shape[0]):
            for layer in range(self.numOfLayers):
                if layer==0:
                    lstmInputs = np.concatenate((inputs[timestep:timestep+1,:],self.stm[layer]),axis=-1)
                else:
                    lstmInputs = np.concatenate((self.stm[layer-1],self.stm[layer]),axis=-1)
                nonActivatedGateOutputs = np.matmul(lstmInputs,self.weights[layer]) + self.biases[layer]
                self.ltm[layer] = self.sigmoid(nonActivatedGateOutputs[...,0*self.structure[layer]:1*self.structure[layer]])*self.ltm[layer] + self.sigmoid(nonActivatedGateOutputs[...,1*self.structure[layer]:2*self.structure[layer]])*np.tanh(nonActivatedGateOutputs[...,2*self.structure[layer]:3*self.structure[layer]])
                self.stm[layer] = self.sigmoid(nonActivatedGateOutputs[...,3*self.structure[layer]:4*self.structure[layer]])*np.tanh(self.ltm[layer])
            output = np.tanh(np.matmul(self.stm[layer],self.weights[-1])+self.biases[-1])
            self.outputs.append(output)
        self.outputs = np.concatenate(self.outputs)
        return self.outputs