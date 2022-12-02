from sys import modules
import numpy as np

class Net:
    def __init__(self, dimensions, activation_funcs, loss_func):
        self.layers = len(dimensions) - 1
        self.loss_func = loss_func
        self.W = {}
        self.b = {x + 1: np.zeros((dimensions[x + 1], 1)) for x in range(self.layers)}
        self.g = {}
        num_neurons = {x + 1: dimensions[x + 1] for x in range(self.layers)}
