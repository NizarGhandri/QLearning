#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 18:09:22 2019

@author: nizar
"""
from Base import BaseQLearning as b
import numpy as np
import sys
import math

class Zap_Q (b): 
    #given we are coding basis functions we take our basis functions to be the indicator functions 1{x = x^n, u = u^n}
    #Thus our Q values will be the stochastically aproximated theta
    
    
    def __init__ (self, state_space, action_space, beta, lambd, theta_0, A_0, starting_state, p_zap_gain): 
        self.beta = beta
        self.lambd = lambd
        self.state_space = state_space
        self.action_space = action_space
        self.size = self.state_space*self.action_space
        self.theta = np.array(theta_0).reshape(self.size,1)
        self.A = np.array(A_0)
        self.basis_f = np.eye(self.size)
        self.swig = self.basis_f[starting_state*self.action_space+self.action(starting_state), :].reshape(self.size, 1)
        self.p = p_zap_gain
        self.n = 0
        
        
    def action(self, s):
        S = s*self.action_space
        return np.argmax(self.theta[S:S+self.action_space, 0])
    
    def update(self, s, s_, action, reward):
        S = s*self.action_space
        S_ = s_*self.action_space
        futur_action = self.action(s_)
        d_n_1 = reward + self.beta * self.theta[S_ + futur_action,0] - self.theta[S + action,0]
        A_n_1 = self.swig@(self.beta*self.basis_f[S_ + futur_action,:].reshape(1, self.size)-self.basis_f[S + action,:].reshape(1, self.size))
        self.A = self.A + (A_n_1 - self.A)*self.zap_gain_n() 
        self.theta = self.theta - self.alpha_n()*np.linalg.pinv(self.A)@self.swig*d_n_1
        self.swig = self.lambd*self.beta*self.swig + self.basis_f[S_ + futur_action,:].reshape(self.size, 1)
        self.n += 1
        
        
    
             
    def zap_gain_n(self):
        return (self.n+1)**self.p
    
    def alpha_n(self):
        return 1/(self.n+1)