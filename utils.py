# utils.py
# utils for deep Boltzmann machine
# Alan Morningstar
# March 2017


import numpy as np


# sigmoid activation function
def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


# sample neuron activation probabilities
def BernoulliSample(x):
    return (x > np.random.uniform(0.0,1.0,x.shape)).astype(int)


# neuron activation
def neuron(x):
    return BernoulliSample(sigmoid(x))
