#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 14:34:15 2019

@author: nizar
"""
import numpy as np

############################### Network Class ###############################

class Network:
    def __init__(self, dimensions, activations, back_propagation_function):
        """
        :param dimensions: list of dimensions of the neural net. (input, hidden layer, ... ,hidden layer, output)
        :param activations: list of activation functions. Must contain N-1 activation function, where N = len(dimensions).
        
        Example of one hidden layer with
        - 2 inputs
        - 10 hidden nodes
        - 5 outputs
        layers -->    [1,        2,          3]
        ----------------------------------------
        dimensions =  (2,     10,          5)
        activations = (      Sigmoid,      Sigmoid)
        """
        
        self.n_layers = len(dimensions)
        self.loss = None
        self.learning_rate = None
        
        # Weights and biases are initiated by index.
        self.w = {}
        self.b = {}

        # Activations are also initiated by index. 
        self.activations = {}
        for i in range(len(dimensions) - 1):
            self.w[i + 1] = np.random.randn(dimensions[i], dimensions[i + 1]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros(dimensions[i + 1])
            self.activations[i + 2] = activations[i]
            
        # Back propagtion function.
        self.back_prop = back_propagation_function

    def feed_forward(self, x):
        """
        Execute a forward feed through the network.
        :param x: (array) Batch of input data vectors.
        :return: (tpl) Node outputs and activations per layer. The numbering of the output is equivalent to the layer numbers.
        """

        # w(x) + b
        z = {}

        # activations: f(z)
        a = {1: x}  # First layer has no activations as input, so we consider input itself as the first activation.
        
        for i in range(1, self.n_layers):
            # current layer = i
            # activation layer = i + 1
            z[i + 1] = a[i]@self.w[i] + self.b[i]
            a[i + 1] = self.activations[i+1](z[i+1])

        return z, a
    
    def predict(self, x):
        """
        :param x: (array) Containing parameters
        :return: (array) A 2D array of shape (n_cases, n_classes).
        """
        _, a = self.feed_forward(x)
        return a[self.n_layers]
    

# =============================================================================
#     def back_prop(self, z, a, y_true):
#         """
#         The input dicts keys represent the layers of the net.
#         a = { 1: x,
#               2: f(w1(x) + b1)
#               3: f(w2(a2) + b2)
#               }
#         :param z: (dict) w(x) + b
#         :param a: (dict) f(z)
#         :param y_true: (array) One hot encoded truth vector.
#         :return:
#         """
# 
#         # Determine partial derivative and delta for the output layer.
#         y_pred = a[self.n_layers]
#         delta = self.loss_function.gradient(y_true, y_pred) * self.activations[self.n_layers].gradient(y_pred)
#         
#         dw = np.dot(a[self.n_layers - 1].T, delta)
# 
#         update_params = {
#             self.n_layers - 1: (dw, delta)
#         }
# 
#         # Determine partial derivative and delta for the rest of the layers.
#         # Each iteration requires the delta from the previous layer, propagating backwards.
#         for i in reversed(range(2, self.n_layers)):
#             delta = np.dot(delta, self.w[i].T) * self.activations[i].gradient(z[i])
#             dw = np.dot(a[i - 1].T, delta)
#             update_params[i - 1] = (dw, delta)
#         
#         # finally update weights and biases
#         for k, v in update_params.items():
#             self.update_w_b(k, v[0], v[1])
# =============================================================================

# =============================================================================
#     def update_w_b(self, index, dw, delta):
#         """
#         Update weights and biases.
#         :param index: (int) Number of the layer
#         :param dw: (array) Partial derivatives
#         :param delta: (array) Delta error.
#         """
# 
#         self.w[index] -= self.learning_rate * dw
#         self.b[index] -= self.learning_rate * np.mean(delta, 0)
# =============================================================================

# =============================================================================
#     def fit(self, x, y_true, loss, epochs, batch_size):
#         """
#         :param x: (array) Containing parameters
#         :param y_true: (array) Containing one hot encoded labels.
#         :param loss: Loss class (MSE, CrossEntropy etc.)
#         :param epochs: (int) Number of epochs.
#         :param batch_size: (int)
#         :param learning_rate: (flt)
#         """
#         if not x.shape[0] == y_true.shape[0]:
#             raise ValueError("Length of x and y arrays don't match")
#             
#         # Initiate the loss object with the final activation function
#         self.loss_function = loss
# 
#         for i in range(epochs):
#             # Shuffle the data
#             indices = np.arange(x.shape[0])
#             np.random.shuffle(indices)
#             x_ = x[indices]
#             y_ = y_true[indices]
# 
#             for j in range(x.shape[0] // batch_size):
#                 k = j * batch_size
#                 l = (j + 1) * batch_size
#                 z, a = self.feed_forward(x_[k:l])
#                 self.back_prop(z, a, y_[k:l])
# 
#             if (i + 1) % 10 == 0:
#                 _, a = self.feed_forward(x)
#                 print("Loss at epoch {}: {}".format(i + 1, self.loss_function.loss(y_true, a[self.n_layers])))
#                 
# =============================================================================
