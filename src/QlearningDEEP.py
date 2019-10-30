#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 16:35:15 2019

@author: nizar
"""

import gym
import numpy as np 
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

#------- Q_Learning hyperparameters --------------
env = gym.make('CartPole-v0')
epsilon = 0.2
epsilon_decay = 0.99
gamma = 0.9

#------- Target neural network --------------------
targetNN = Sequential()
targetNN.add(Dense(5, input_shape = (5,) , activation= 'relu'))
targetNN.add(Dense(16, activation= 'relu'))
targetNN.add(Dense(1, activation= 'linear'))
targetNN.compile(optimizer='adam', loss='mean_squared_error')
print(targetNN.summary())


#-------- Prediction neural network ---------------
predictionNN = Sequential()
predictionNN.add(Dense(5, input_shape = (5,), activation= 'relu'))
predictionNN.add(Dense(16, activation = 'relu'))
predictionNN.add(Dense(1, activation= 'linear'))
predictionNN.compile(optimizer='adam', loss='mean_squared_error')
print(predictionNN.summary())

#-------- Updating for same parameters -------------
targetNN.set_weights(predictionNN.get_weights())


#-------- Learning parameters ----------------------
C = 5
replay_buffer = []
rb_index = 0
max_correction = np.zeros(5)
batch = np.zeros((5,5))

for i_episode in range(700):
    observation = env.reset()
    for t in range(100):
        env.render()
        if (random.uniform(0, 1) < epsilon):
            action = random.randint(0, 1) 
        else: 
            x,y,z,c = observation
            a_0 = np.array([0 , x , y, z, c]).reshape(1,5)
            a_1 = np.array([1 , x , y, z, c]).reshape(1,5)
            if(predictionNN.predict(a_0) > predictionNN.predict(a_1)):
                action = 0
            else: 
                action = 1
        observationNew, reward, done, info = env.step(action)
        replay_buffer.append((reward, observationNew, observation, action))
        rb_index = (rb_index + 1)%100
        for i in range(5):
            k = random.randint(0, rb_index) 
            r,oN,o,a = replay_buffer[0]
            t_0, t_1 = np.array([0, oN[0], oN[1], oN[2], oN[3]]).reshape(1,5), np.array([1, oN[0], oN[1], oN[2], oN[3]]).reshape(1,5)
            max_correction[i] = r + gamma*max(targetNN.predict(t_1), targetNN.predict(t_0))
            batch[i][0] = a
            w,d,f,g = o
            batch[i][1] = w
            batch[i][2] = d
            batch[i][3] = f
            batch[i][4] = g
            
        predictionNN.fit(batch, max_correction)
        epsilon = epsilon_decay*epsilon
        if(len(replay_buffer) % 5 == 0):
            targetNN.set_weights(predictionNN.get_weights())
        #q_table[state, action] = q_table[state, action] + lr * (reward + gamma * np.max(q_table[new_state, :]) — q_table[state, action])
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()




def step(x): 
    if(x < 0.5):
        return 0
    return 1