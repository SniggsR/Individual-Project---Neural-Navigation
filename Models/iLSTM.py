# -*- coding: utf-8 -*-
"""
Created on Sun Jan 23 13:37:22 2022

@author: Kishan
"""

import tensorflow as tf
import numpy as np

class iLSTM(tf.keras.layers.Layer):
    def __init__(self, numOfUnits):
        super(iLSTM,self).__init__(name="iLSTM")
        self.numOfUnits = numOfUnits
        pass
    
    def build(self, input_shape):
        self.w = tf.Variable(tf.random.normal([input_shape[-1]+self.numOfUnits, 4*self.numOfUnits]), trainable=True, name='weightKernel', dtype=tf.float32)
        self.b = tf.Variable(tf.zeros([1, 4*self.numOfUnits]), trainable=True, name='biasKernel', dtype=tf.float32)
        pass
    
    def call(self, inputs, shortTermMemory, longTermMemory):
        concatenatedInputs = tf.concat([inputs, shortTermMemory], axis=-1) #a.k.a gate inputs
        nonActivatedGateOutputs = tf.einsum('bti,iu -> btu', concatenatedInputs, self.w) + self.b
        longTermMemory = tf.sigmoid(nonActivatedGateOutputs[...,0*self.numOfUnits:1*self.numOfUnits])*longTermMemory + tf.sigmoid(nonActivatedGateOutputs[...,1*self.numOfUnits:2*self.numOfUnits])*tf.tanh(nonActivatedGateOutputs[...,2*self.numOfUnits:3*self.numOfUnits])
        shortTermMemory = tf.sigmoid(nonActivatedGateOutputs[...,3*self.numOfUnits:4*self.numOfUnits])*tf.tanh(longTermMemory)
        return shortTermMemory, longTermMemory

class core(tf.keras.Model):
    def __init__(self, structure):
        super(core,self).__init__(name="iLSTMCore")
        self.structure = structure
        pass
    
    def build(self, input_shape):
        self.Layers = [iLSTM(numOfUnits) for _, numOfUnits in enumerate(self.structure)]
        pass
    
    def createMemory(self, input_shape):
        shortTermMemory = np.asarray(np.zeros([input_shape[0],1,sum(self.structure)]), dtype=np.float32)
        longTermMemory = np.asarray(np.zeros([input_shape[0],1,sum(self.structure)]), dtype=np.float32)
        return shortTermMemory, longTermMemory
    
    def call(self, inputs, shortTermMemory, longTermMemory):
        shortTermMemory = tf.split(shortTermMemory, self.structure, axis=-1)
        longTermMemory = tf.split(longTermMemory, self.structure, axis=-1)
        for id, numOfUnits in enumerate(self.structure):
            if id == 0:
                shortTermMemory[id], longTermMemory[id] = self.Layers[id](inputs, shortTermMemory[id], longTermMemory[id])
            else:
                shortTermMemory[id], longTermMemory[id] = self.Layers[id](shortTermMemory[id-1], shortTermMemory[id], longTermMemory[id])
        outputs = shortTermMemory[id]
        shortTermMemory = tf.concat(shortTermMemory,axis=-1)
        longTermMemory = tf.concat(longTermMemory,axis=-1)
        return outputs, shortTermMemory, longTermMemory