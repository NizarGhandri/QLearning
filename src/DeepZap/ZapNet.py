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
    
    def __init__(self, dimensions, self.beta=0.99, self.zapgain=0.999999999999999999999999999999999999999999999, ):
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
        
        
        
    def update(self, s, s_, action, reward):
        def projection (x):
            if(x < 0.0001 and x > -0.0001):  
                return np.sign(x)*0.0001
        S = s*self.action_space
        S_ = s_*self.action_space
        futur_action = self.action(s_)
        self.swig = self.basis_f[S + action,:].reshape(self.size, 1)
        d_n_1 = reward + self.beta * self.theta[S_ + futur_action,0] - self.theta[S + action,0]
        A_n_1 = self.swig@(self.beta*self.basis_f[S_ + futur_action,:].reshape(1, self.size)-self.basis_f[S + action,:].reshape(1, self.size))
        #projection_vectorized = np.vectorize(projection)
        self.A = (self.A + (A_n_1 - self.A)*self.zapgain) 
        #self.A = projection_vectorized(self.A)
       # if(self.n == 5000):
        #    print(self.A)
         #   sys.exit(0)
        #l = np.transpose(self.A)
        #print(self.A + 0.0000000000001 * np.eye(self.size))
        G = -np.linalg.pinv(self.A + 0.00000000000001 * np.eye(self.size))
        self.theta = self.theta + self.alpha_n()*G@self.swig*d_n_1
        #self.alpha_n()
        self.n += 1
        
    def reward(x, y, loss, lmbd=0.1):
            return np.exp(lmbd*loss(x,y))/lmbd
        
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