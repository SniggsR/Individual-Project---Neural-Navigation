# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 14:33:04 2022

@author: Kishan
"""

import numpy as np

class linearRegressionInference:
    def __init__(self, weights, biases, structure = [10,10], activation='tanh'):
        self.weights = weights
        self.biases = biases
        self.activation = activation
        self.structure = structure
        self.numOfInputs = np.int(self.weights[0].shape[0])
        self.numOfOutputs = np.int(self.biases[-1].shape[-1])
        pass
    
    def sigmoid(self,inputs):
        return 1/(1+np.exp(-inputs))
    
    def infer(self,inputs):
        if self.activation == 'tanh':
            self.outputs = np.tanh(np.matmul(inputs,self.weights[0])+self.biases[0])
            for layer in range(len(self.structure)-1):
                self.outputs = np.tanh(np.matmul(self.outputs,self.weights[layer+1])+self.biases[layer+1])
        elif self.activation == 'sigmoid':
            self.outputs = self.sigmoid(np.matmul(inputs,self.weights[0])+self.biases[0])
            for layer in range(len(self.structure)-1):
                self.outputs = self.sigmoid(np.matmul(inputs,self.weights[layer+1])+self.biases[layer+1])
        else:
            self.outputs = np.matmul(inputs,self.weights[0])+self.biases[0]
            for layer in range(len(self.structure)-1):
                self.outputs = np.matmul(inputs,self.weights[layer+1])+self.biases[layer+1]
        return self.outputs