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
        self.theta = np.array(theta_0)
        self.A = np.array(A_0)
        self.state_space = state_space
        self.action_space = action_space
        self.basis_f = np.eye(self.state_space*self.action_space)
        self.swig = self.basis_f[:, starting_state*self.action_space+self.action(starting_state)]
        self.p = p_zap_gain
        self.n = 0
        
        
    def action(self, s):
        S = s*self.action_space
        return np.argmax(self.theta[S:S+self.action_space])
    
    def update(self, s, s_, action, reward):
        S = s*self.action_space
        S_ = s_*self.action_space
        futur_action = self.action(s_)
        d_n_1 = reward + self.beta * self.theta[S_ + futur_action] - self.theta[S + action]
        A_n_1 = self.swig@(self.beta*self.basis_f[S_ + futur_action,:]-self.basis_f[S + action,:])
        self.A = self.A + (A_n_1 - self.A)*self.zap_gain_n() 
        print(self.theta)
        self.theta = self.theta - np.transpose(self.alpha_n()*np.linalg.pinv(self.A)@self.swig*d_n_1)
        print(self.theta)
        print(np.transpose(self.alpha_n()*np.linalg.pinv(self.A)@self.swig*d_n_1))
        if(math.isnan(self.theta[0])):
            sys.exit(0)
        self.swig = self.lambd*self.beta*self.swig + self.basis_f[:, S_ + futur_action]
        self.n += 1
             
    def zap_gain_n(self):
        return (self.n+1)**self.p
    
    def alpha_n(self):
        return 1/(self.n+1)