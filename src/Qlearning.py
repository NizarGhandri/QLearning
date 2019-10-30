#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:19:05 2019

@author: nizar
"""
import gym
import random
import numpy as np
from Tabular import Tabular
import matplotlib.pyplot as plt

Modes_Available = {'Tabular': lambda sp, ap, lr, d, lr_d, d_d : Tabular(sp, ap, lr, d, lr_d, d_d)}

def Qlearning (mode, environment, number_of_episodes, timestep_per_episode, learning_rate, discount, epsilon, epsilon_min, learning_rate_decay, discount_decay):
    
    env = gym.make(environment)
    QlearningObject = Modes_Available[mode](env.observation_space.n, env.action_space.n, learning_rate, discount, learning_rate_decay, discount_decay)
    e = epsilon
    rewards = []
    for i_episode in range(number_of_episodes):
        r = 0
        observation = env.reset()
        for t in range(timestep_per_episode):
            env.render()
            if (random.uniform(0, 1) < e):
                action = env.action_space.sample()
            else:
                action = QlearningObject.action(observation)
            observationN, reward, done, info = env.step(action)
            QlearningObject.update(observation, observationN, action, reward)
            observation = observationN
            r += reward
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
        rewards.append(r)
        e = epsilon_min + (1 - epsilon_min) * np.exp(-1/number_of_episodes*i_episode)
    print(QlearningObject.QTable)
    f = np.split(np.array(rewards), 10)
    plotted = []
    for r in f: 
        z = sum(r)/1000
        print(z)
        plotted.append(z)
    plt.plot(plotted)
    plt.show()
    env.close()
    
    
    