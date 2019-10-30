#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:19:05 2019

@author: nizar
"""
import gym
import numpy as np 
import random
from Tabular import Tabular

Modes_Available = {'Tabular': lambda x : x}

def Qlearning (learning_rate, discount, epsilon, mode, environment, number_of_episodes, timestep_per_episode, epsilon_decay, learning_rate_decay):
    
    QlearningObject = Modes_Available(mode)(discount)
    QlearningObject.setLearningRate(learning_rate), QlearningObject.setLearningRate(learning_rate)
    env = gym.make(environment)
    for i_episode in range(number_of_episodes):
        observation = env.reset()
        for t in range(timestep_per_episode):
            env.render()
            print(observation)
            if (random.uniform(0, 1) < epsilon):
                action = random.randint(0, 1)
            else:
                action = QlearningObject.action(observation)
            observation, reward, done, info = env.step(action)
            QlearningObject.update()
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()
    
    
    
