#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:56:05 2019

@author: nizar
"""
import numpy as np

############################### Activation functions ###############################


def linear(x):
    return x


def sigmoid(x):
    return 1/(1+np.exp(-x))