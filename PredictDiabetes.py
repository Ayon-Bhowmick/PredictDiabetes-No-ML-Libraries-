import pickle
import numpy as np
from modules.net import Net
from modules.funcs import Sigmoid, ReLU, CrossEntropyLoss
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    """
    load and split data
    """
    with open("data\\data.pkl", "rb") as f:
        X, y = pickle.load(f)
    y = np.eye(2)[y.astype(int)]
    n = X.shape[0]
    n_train = int(n * 0.8)
    idx = np.random.permutation(n)
    X_train, y_train = X[idx[:n_train],], y[idx[:n_train]]
    X_test, y_test = X[idx[n_train:],], y[idx[n_train:]]
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = load_data()

    kwargs = {
        "training X": X_train,
        "training y": y_train,
        "test X": X_test,
        "test y": y_test,
        "max_iters": 201,
        "learning rate": 0.001,
        "weight decay": 0.0001,
        "mini-batch size": 50,
        "record_every": 10,
        "test loss": "0-1 error"
    }

    dimensions = [X_train.shape[0], 64, 64, y_train.shape[0]]
    activation = {1: ReLU, 2: ReLU, 3: Sigmoid}
    loss = CrossEntropyLoss
    nn = Net(dimensions, activation, loss)

    nn.train(**kwargs)
    predicted = np.argmax(nn.forward(kwargs['test X']), axis=0)
    truth = np.argmax(kwargs['test y'], axis=0)
