#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:09:02 2019

@author: nizar
"""
import QlearningFrozen 

print("-----------------Trial for 4 * 4----------------")

QlearningFrozen.Qlearning('Tabular', 'FrozenLake-v0', 10000, 100, 0.1, 0.9, 1, 0.1, 1, 1) ###### goes up to 0.9 average reweard on the last 1000 runs 

print("-----------------Trial for 8 * 8----------------")

QlearningFrozen.Qlearning('Tabular', 'FrozenLake8x8-v0', 100000, 1000, 0.1, 0.9, 1, 0.1, 1, 1) ####### doesn't work with the same number of episodes but if u 
                                                                                               ###### multiply it by 10 you can reach 0.5 average rewards on the last 1000 runs