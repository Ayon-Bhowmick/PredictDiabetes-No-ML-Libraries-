from sys import modules
import numpy as np

class Net:
    def __init__(self, dimensions, activation_funcs, loss_func):
        """
        L-layer feed forward network.
        """
        self.layers = len(dimensions) - 1
        self.loss_func = loss_func

        self.W = {}
        self.b = {}
        self.g = {}
        num_neurons = {}
        for l in range(self.num_layers):
            num_neurons[l + 1] = dimensions[l + 1]
            nin, nout = dimensions[l], dimensions[l + 1]
            sd = np.sqrt(2.0 / (nin + nout))
            self.W[l + 1] = np.random.normal(0.0, sd, (nout, nin))
            self.b[l + 1] = np.zeros((dimensions[l + 1], 1))
            self.g[l + 1] = activation_funcs[l + 1]

        self.A = {}
        self.Z = {}
        self.dZ = {}
        self.dW = {}
        self.db = {}

    def forward(self, X):
        """
        forward computation of activations at each layer.
        """
        self.A[0] = X
        for l in range(1, self.num_layers + 1):
            self.Z[l] = self.W[l] @ self.A[l - 1] + self.b[l]
            self.A[l] = self.g[l].activate(self.Z[l])
        return self.A[l]

    def backward(self, Y):
        """
        back propagation to compute the gradients of parameters at all layers.
        """
        B = 1 / Y.shape[1]

        self.dZ[self.num_layers] = self.A[self.num_layers] - Y
        self.db[self.num_layers] = ((B * np.ones((1, Y.shape[1]))) @ self.dZ[self.num_layers].T).T
        self.dW[self.num_layers] =  B * (self.A[self.num_layers] - Y) @ self.A[self.num_layers - 1].T

        for l in range(self.num_layers - 1, 0, -1):
            self.dZ[l] = np.multiply((self.W[l + 1].T * self.dZ[l + 1]), self.g[l].gradient(self.Z[l]))
            self.db[l] = ((B * np.ones((1 ,Y.shape[1]))) * self.dZ[l].T).T
            self.dW[l] = B * self.dZ[l] * self.A[l - 1].T

        return self.dW, self.db

    def update_parameters(self, lr, weight_decay = 0.001):
        for l in range(1, self.num_layers + 1):
            self.W[l] -= lr * self.dW[l] + weight_decay * self.W[l]
            self.b[l] -= lr * self.db[l]

    def train(self, **kwargs):
        """
        mini-batch stochastic gradient descent
        """
        X_train = kwargs['Training X']
        Y_train = kwargs['Training Y']
        num_samples = X_train.shape[1]
        iter_num = kwargs['max_iters']
        lr = kwargs['Learning rate']
        weight_decay = kwargs['Weight decay']
        batch_size = kwargs['Mini-batch size']
        record_every = kwargs['record_every']

        for it in range(iter_num):
            for i in range(0, num_samples, batch_size):
                X = X_train[:, i:i + batch_size]
                y = Y_train[:, i:i + batch_size]
                self.forward(X)
                self.backward(y)
                self.update_parameters(lr, weight_decay)

        if (it + 1) % record_every == 0:
                if 'Test X' in kwargs and 'Test Y' in kwargs:
                   prediction_accuracy = self.test(**kwargs)
                   print(', test error = {}'.format(prediction_accuracy))

    def test(self, **kwargs):
        """
        test accuracy of the trained model
        """
        X_test = kwargs['Test X']
        Y_test = kwargs['Test Y']
        loss_func = kwargs['Test loss function name']
        output = self.forward(X_test)

        if loss_func == '0-1 error':
            predicted_labels = np.argmax(output, axis = 0)
            true_labels = np.argmax(Y_test, axis = 0)
            return 1.0 - accuracy_score(np.array(true_labels).flatten(), np.array(predicted_labels).flatten())
        else:
            return np.linalg.norm(output - Y_test) ** 2 / (2 * Y_test.shape[1])

def accuracy_score(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)
