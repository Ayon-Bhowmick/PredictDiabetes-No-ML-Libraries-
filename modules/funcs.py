import numpy as np
from scipy.special import xlogy

class Sigmoid:
    @staticmethod
    def activate(Z):
        """
        Sigmoid of each element of Z.
        """
        return 1 / (1 + np.exp(-Z))

    @staticmethod
    def gradient(Z):
        """
        Gradient of sigmoid at Z
        """
        for i in range(Z.shape[0]):
            for j in range(Z.shape[1]):
                Z[i, j] = Sigmoid.activate(Z[i, j]) * (1 - Sigmoid.activate(Z[i, j]))
        return Z

class ReLU:
    @staticmethod
    def activate(Z):
        """
        ReLu of each element of Z.
        """
        return np.maximum(0, Z)

    @staticmethod
    def gradient(Z):
        """
        Gradient of ReLU at Z
        """
        return (Z > 0).astype(int)

class CrossEntropyLoss:
    @staticmethod
    def loss(Y, Y_hat):
        """
        Averaged cross entropy loss
        """
        return -np.mean(np.sum(xlogy(Y, Y_hat), axis=0))

    @staticmethod
    def gradient(Y, Y_hat):
        """
        Gradients on the training examples
        """
        return Y_hat - Y
