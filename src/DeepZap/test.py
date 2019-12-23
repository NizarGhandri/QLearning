#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 14:05:15 2019

@author: nizar
"""

from ZapNet import ZapNet


NN = ZapNet([4, 7, 4, 2], 'CartPole-v0')
NN.fit(5000, 100)
NN.evaluate(1000, 100)