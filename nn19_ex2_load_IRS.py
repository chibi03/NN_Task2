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

plt.figure()
plt.xticks(np.arange(0, 19, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel('accuracy')
plt.xlabel('features')
plt.title('IRLS accuracy vs number of features')
plt.ylim([0, 1])


def plot_point_values(x_y_array, ax, label):
    for row in x_y_array:
        valign = 'bottom'
        offset_x = row[0]-0.05
        offset = row[1]-0.02
        if label == 'train':
            valign = 'top'
            offset_x = row[0]+0.05
            offset = row[1]+0.02
        ax.annotate('{:.0%}'.format(row[1]), xy=(row[0], row[1]), xytext=(offset_x, offset),
                    ha='left', va=valign)


def nabla_e(X, y, target):
    return X @ (y - target)


def hessian(X, y):
    R = np.diag(y[0]*(1-y[0]))
    return X.T @ R @ X


def update(weights, hessian, gradient):
    return weights - (np.linalg.inv(hessian) @ gradient)


def log_likelihood(estimate, target):
    cls_1 = target
    cls_2 = (1-target)
    err_per_example = np.where(cls_1 != 0, cls_1 * np.log(estimate), 0)
    err_per_example = np.where(cls_2 != 0, err_per_example + cls_2 * np.log(1 - estimate), 0)
    return -np.sum(err_per_example, axis=0)


def irls(X, target, flag="train"):
    result = np.zeros((X.shape[1]-2, 2), dtype=float)
    for features in range(2, X.shape[1]):
        bias = np.ones((X.shape[0], 1))
        input = np.append(bias, X[:, 0:features], axis=1)
        weights = np.random.uniform(low=-0.001, high=0.001, size=input.shape[1])[:, np.newaxis]
        y = sigmoid(weights.reshape((1, -1)) @ input.T)  # probability of class 1

        error_old = 100
        error_new = 99

        print("--- Computing IRLS with ", features, " features.")
        while (error_old - error_new) > 0.1:
            error_old = error_new

            # update until error converges
            hess = hessian(input, y)
            gradient = nabla_e(input.T, y.reshape((-1, 1)), target)
            weights = update(weights, hess, gradient)

            y = sigmoid(weights.reshape((1, -1)) @ input.T)
            error_new = log_likelihood(y.reshape((-1, 1)), target)
            print("error:", error_new)

        probability = np.where(y > 0.5, 1, 0)
        comparison = np.equal(probability.reshape((-1, 1)), target)
        accuracy = np.count_nonzero(comparison)/target.size
        print(flag, "---Nr of features ", features, " accuracy", accuracy)

        # plot graph
        result[features-2][0] = features
        result[features-2][1] = accuracy

    plt.plot(result[:, 0], result[:, 1], '+-', label=flag)
    plot_point_values(result, plt.gca(), flag)


irls(X, np.where(C == 2, 0, 1))
irls(Xtst, np.where(Ctst == 2, 0, 1), 'test')

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.show()

