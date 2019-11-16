#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 20:52:37 2019

@author: nizar
"""

import numpy as np 
from Base import BaseQLearning


class Tabular(BaseQLearning): 
    
    def __init__ (self, state_space, action_space, learning_rate, discount, learning_rate_decay, discount_decay): 
        self.QTable = np.zeros((state_space, action_space))
        self.alpha = learning_rate
        self.gamma = discount 
        self.alpha_decay = learning_rate_decay
        self.gamma_decay = discount_decay
        self.n = 0
        
    
    def action (self, s): #state
        return np.argmax(self.QTable[s,:])
    
    def update (self, s, s_, action, reward): 
        self.QTable[s, action] = self.QTable[s, action] + self.alpha*(reward + self.gamma*np.max(self.QTable[s_,:]) - self.QTable[s, action])
        
    