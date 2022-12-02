import csv
import numpy as np
import pickle

with open("..\\data\\diabetes.csv", "r") as f:
    reader = csv.reader(f)
    data: np.ndarray = np.array(list(reader))[1:,]
y: np.ndarray[float] = data[:, -1].astype(float)
X: np.ndarray[float] = data[:, :-1].astype(float)

with open("..\\data\\data.pkl", "wb") as f:
    pickle.dump((X, y), f)
