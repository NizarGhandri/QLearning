#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 22:10:40 2019

@author: nizar
"""

from abc import ABC, abstractmethod

class BaseQLearning (ABC):
    @abstractmethod
    def update ():
        pass
    
    @abstractmethod
    def action ():
        pass
    
    @abstractmethod
    def temp ():
        pass
    