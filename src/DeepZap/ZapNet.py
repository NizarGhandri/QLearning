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
from Network import Network
import gym
import matplotlib.pyplot as plt
import sys

class ZapNet: 
    
    def __init__(self, dimensions, environment, beta=0.99, zapgain=0.999999999999999999999999999999999999999999999, epsilonzap=0.00000000000001):
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
        self.dimensions = dimensions
        activations =np.full(len(dimensions)-1, ActivationFunctions.linear)
        self.neural_net = Network(dimensions, activations, self.update)
        self.beta = beta
        self.zapgain = zapgain
        self.epsilonzap = epsilonzap
        self.env = gym.make(environment)
        self.n = 0
        self.A = []
        temp = len(dimensions)-1
        for i in range(temp):
            self.A.append([np.zeros((dimensions[i+1], dimensions[i+1])) for j in range(dimensions[i])])
    
    def nlayers (self):
        return self.neural_net.n_layers
        
    def action(self, s):
        return np.argmax(self.neural_net.predict(s))
    
    ###### for now only supports linear activation function    
    def update_layer(self, reward, a, a_, i):
        number_of_neurones = self.dimensions[i]
        number_of_neurones_next_layer = self.dimensions[i+1]
        k = i + 1
        for j in range(number_of_neurones):
            swig = np.full(number_of_neurones_next_layer, a[k][0][j])
            swig_f = np.full(number_of_neurones_next_layer, a_[k][0][j])
            d_n_1 = reward + self.beta * self.neural_net.w[k][j]@swig_f - self.neural_net.w[k][j]@swig
            A_n_1 = swig@(self.beta*swig_f-swig)
            self.A[i][j] = (self.A[i][j] + (A_n_1 - self.A[i][j])*self.zapgain) 
            G = -np.linalg.pinv(self.A[i][j] + self.epsilonzap * np.eye(number_of_neurones_next_layer))
            self.neural_net.w[k][j] = self.neural_net.w[k][j] + self.alpha_n()*G@swig*d_n_1
        
    def update(self, observation, observationN, action, reward):
     
        _, a = self.neural_net.feed_forward(observation)
        _, a_ = self.neural_net.feed_forward(observationN)
        #print(a)
        for i in range(self.nlayers()-2, -1, -1):
            #print(i+1)
            self.update_layer(reward, a, a_, i)
        self.n += 1
        #sys.exit(0)
            
    def fit(self, number_of_episodes, timestep_per_episode):
         e = 0.9
         for i_episode in range(number_of_episodes):
            observation = np.array(self.env.reset()).reshape(1,1)
            for t in range(timestep_per_episode):
                self.env.render()
                if (np.random.uniform() < e):
                    action = self.env.action_space.sample()
                else:
                    action = self.action(observation)
                observationN, reward, done, info = self.env.step(action)
                observationN = np.array(observationN).reshape(1,1)
                self.update(observation, observationN, action, reward)
                observation = observationN
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break



    def evaluate(self, number_of_episodes, timestep_per_episode):
        rewards = []
        self.n = 0
        for i_episode in range(number_of_episodes):
                r = 0
                observation = np.array(self.env.reset()).reshape(1,1)
                for t in range(timestep_per_episode):
                    self.env.render()
                    action = self.action(observation)
                    observationN, reward, done, info = self.env.step(action)
                    observationN = np.array(observationN).reshape(1,1)
                    self.update(observation, observationN, action, reward)
                    observation = observationN
                    r += reward
                    if done:
                        print("Episode finished after {} timesteps".format(t+1))
                        break
                rewards.append(r)
        f = np.split(np.array(rewards), number_of_episodes//100)
        plotted = []
        for r in f: 
            z = sum(r)/100
            print(z)
            plotted.append(z)
        plt.plot(plotted)
        plt.show()
        self.env.close()
            
        
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