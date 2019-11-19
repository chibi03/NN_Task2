# This example script shows how to load the data set in python
# and how to extract the input samples and targets

import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_in = open('vehicle.pkl', 'rb')
vehicle_data = pckl.load(file_in)
file_in.close()

# Training set
X = vehicle_data['train']['X']  # features; X[i,j]...feature j of example i
C = vehicle_data['train']['C']  # classes; C[i]...class of example i
# Test set
Xtst = vehicle_data['test']['X']  # features
Ctst = vehicle_data['test']['C']  # classes

# extract examples of class SAAb (2) and VAN (4)
indices = np.flatnonzero((C == 4) | (C == 2))
C = C[indices]
X = X[indices]

indices_tst = np.flatnonzero((Ctst == 4) | (Ctst == 2))
Ctst = Ctst[indices_tst]
Xtst = Xtst[indices_tst]

sigmoid = lambda a: np.where(a >= 0,
                             1 / (1 + np.exp(-a)),
                             np.exp(a) / (1 + np.exp(a)))


def nabla_e(X, y, target):
    return X @ (y - target)


def hessian(X, y):
    R = np.diag(y[0]*(1-y[0]))
    return X.T @ R @ X


def update(weights, hessian, gradient):
    return weights - np.linalg.inv(hessian) @ gradient


def log_likelihood(estimate, target):
    cls_1 = target
    cls_2 = (1-cls_1)
    err_per_example = np.where(cls_1 != 0, cls_1 * np.log(estimate), 0)
    err_per_example = np.where(cls_2 != 0, err_per_example + cls_2 * np.log(1 - estimate), 0)
    return -np.sum(err_per_example, axis=0)


def irls(X, target):
    for features in range(2, X.shape[0]):
        bias = np.ones((X.shape[0], 1))
        input = np.append(bias, X[:, 0:features], axis=1)
        weights = np.random.normal(0, 2, size=input.shape[1])[:, np.newaxis]
        y = sigmoid(weights.reshape((1, -1)) @ input.T)

        iteration = 0
        max_iterations = 100

        print("--- Computing IRLS with ", features, " features.")
        while iteration < max_iterations:
            # update until error converges
            hess = hessian(input, y)
            gradient = nabla_e(input.T, y.reshape((-1, 1)), target)
            weights = update(weights, hess, gradient)

            y = sigmoid(weights.reshape((1, -1)) @ input.T)
            error = log_likelihood(y.reshape((-1, 1)), np.where(C == 2, 0, 1))
            print("iteration: ", iteration, "error:", error)
            iteration += 1


irls(X, np.where(C == 2, 0, 1))
