#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 15:31:30 2019

@author: nizar
"""

############################### Zap Net ###############################

# The following class is a decorator of neural network previously coded

import numpy as np
import ActivationFunctions 
import Network

class ZapNet: 
    
    def __init__(self, dimensions, beta=0.99, zapgain=0.999999999999999999999999999999999999999999999, epsilonzap=0.00000000000001):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [1,        2,          3]
        
        activation in this case will always be linear until the output layer where it will be a sigmoid. 
        
        the back propagation will be done in the Zap update manner
        """
        activations = np.append(np.full(len(dimensions)-2, ActivationFunctions.linear), ActivationFunctions.sigmoid)
        self.neural_net = Network(dimensions, activations, self.update)
        self.beta = beta
        self.zapgain = zapgain
        self.epsilonzap = epsilonzap
    
    def nlayers (self):
        return self.neural_net.nlayers
        
    def action(self, s):
        return np.argmax(self.neural_net.predict(s))
        
    def update_layer(self, reward, a, a_, i):
        number_of_neurones = self.neural_net.dimensions[i]
        number_of_neurones_next_layer = self.neural_net.dimensions[i+1]
        for j in range(number_of_neurones):
            swig = np.full(number_of_neurones_next_layer, self.neural_net.a[i, 0])
            swig_f = np.full(number_of_neurones_next_layer, self.neural_net.a_[i, 0])
            d_n_1 = reward + self.beta * self.neural_net.w[i,j]@swig_f - self.neural_net.w[i,j]@swig
            A_n_1 = swig@(self.beta*self.swig_f-swig)
            self.A = (self.A + (A_n_1 - self.A)*self.zapgain) 
            G = -np.linalg.pinv(A + self.epsilonzap * np.eye(number_of_neurones_next_layer))
            self.neural_net.w[i,j] = self.neural_net.w[i,j] + self.alpha_n()*G@swig*d_n_1
        
    def update(self, reward, a, a_):
        for i in range(self.nlayers()-1, 0, -1):
            self.update_layer(reward, a, a_, i)
            
    
        
# =============================================================================
#     def reward(x, y, loss, lmbd=0.1):
#             return np.exp(lmbd*loss(x,y))*lmbd
# =============================================================================
        
    def zap_gain_n(self):
# =============================================================================
#         if(self.training):
#             return 1
#         else:
# =============================================================================
        #print("zappy", 1000*(self.n+1001)**self.p)
        return 250/((self.n)**0.85 + 1000)
    
    
    def alpha_n(self):
# =============================================================================
#         if(self.training):
#             return 0.5
#         else:
# =============================================================================
        #print(1000*(self.n+1001)**-1)
        return 500/(self.n +1001)