import pickle
import numpy as np
from modules.net import Net
from modules.funcs import Sigmoid, ReLU, CrossEntropyLoss

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
    print(y_test)
    return X_train, y_train, X_test, y_test

load_data()
