#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:09:02 2019

@author: nizar
"""
import QlearningFrozen 

print("-----------------Trial for 4 * 4----------------")

QlearningFrozen.Qlearning('Tabular', 'FrozenLake-v0', 10000, 100, 0.1, 0.9, 1, 0.1, 1, 1)
QlearningFrozen.Qlearning('Tabular', 'FrozenLake8x8-v0', 10000, 1000, 0.1, 0.9, 1, 0.1, 1, 1)