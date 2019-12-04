#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 17:19:05 2019

@author: nizar
"""
import gym
import sys
import numpy as np
from Tabular import Tabular
from Zap import Zap_Q
import matplotlib.pyplot as plt


Modes_Available = {'Tabular': lambda sp, ap, lr, d, lr_d, d_d : Tabular(sp, ap, lr, d, lr_d, d_d), 'Zap': lambda state_space, action_space, beta, lambd, theta_0, A_0, starting_state, p_zap_gain: Zap_Q(state_space, action_space, beta, lambd, theta_0, A_0, starting_state, p_zap_gain)}

def Qlearning (mode, environment, number_of_episodes, timestep_per_episode, learning_rate, discount, epsilon, epsilon_min, learning_rate_decay, discount_decay, pzap=-0.85, lambd = 0):
    
    env = gym.make(environment)
    observation = env.reset()
    if(mode != 'Zap'):
        QlearningObject = Modes_Available[mode](env.observation_space.n, env.action_space.n, learning_rate, discount, learning_rate_decay, discount_decay)
    else:
        size = env.observation_space.n*env.action_space.n
        QlearningObject = Modes_Available[mode](env.observation_space.n, env.action_space.n, discount, lambd , np.zeros(size), np.ones((size, size)), observation, pzap)
    e = 0.9
    rewards = []
    
    
    ########## training ############
    for i_episode in range(number_of_episodes):
        observation = env.reset()
        for t in range(timestep_per_episode):
            env.render()
            if (np.random.uniform() < e):
                action = env.action_space.sample()
            else:
                action = QlearningObject.action(observation)
            observationN, reward, done, info = env.step(action)
            QlearningObject.update(observation, observationN, action, reward)
            observation = observationN
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                #if (not(np.array_equal(QlearningObject.theta, np.zeros((env.observation_space.n* env.action_space.n,1))))):
                    #if (QlearningObject.QTable[14,2] < QlearningObject.QTable[14,1]):
                 #    print(QlearningObject.theta)
                  #   sys.exit(0)
                break
       # e = epsilon_min + (epsilon - epsilon_min) * np.exp(-0.007*i_episode)
    print(QlearningObject.theta)    
    #sys.exit(0)  
    ########## evaluation ############    
    QlearningObject.n = 0  
    Qtables = []
    gain_matrices = []
    for i_episode in range(number_of_episodes):
            r = 0
            observation = env.reset()
            for t in range(timestep_per_episode):
                env.render()
                action = QlearningObject.action(observation)
                observationN, reward, done, info = env.step(action)
                QlearningObject.update(observation, observationN, action, reward)
                observation = observationN
                r += reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    #if (not(np.array_equal(QlearningObject.theta, np.zeros((env.observation_space.n* env.action_space.n,1))))):
                        #if (QlearningObject.QTable[14,2] < QlearningObject.QTable[14,1]):
                     #    print(QlearningObject.theta)
                      #   sys.exit(0)
                    break
            rewards.append(r)
            Qtables.append(QlearningObject.theta)
            gain_matrices.append(QlearningObject.A)
        #if(mode == 'Zap'):
         #   QlearningObject.ep()
       # 
   # A = Qtables[::100]
   # for i in range(len(A)):
    #    print("######## Qtable number %d ########" %i)
       # print(A[i])
    f = np.split(np.array(rewards), 300)
    plotted = []
    for r in f: 
        z = sum(r)/100
        print(z)
        plotted.append(z)
    print(-np.linalg.inv((gain_matrices[np.argmax(z)]) + 0.000000000000000000000001*np.eye(size)))
    plt.plot(plotted)
    plt.show()
    env.close()