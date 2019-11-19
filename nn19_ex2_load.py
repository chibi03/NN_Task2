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

# find prior probability
nr_training_examples = C.size

unique, examples_per_class = np.unique(C, return_counts=True)  # counts .. number of examples per class, unique .. classes
prior = examples_per_class / nr_training_examples  # convert count into percentage
priors = dict(zip(unique, prior))

# find mean for each class
mean = {}
i = 0
for cls in unique:
    mean[cls] = (1/examples_per_class[i]) * np.sum(X[np.flatnonzero(C == cls)], axis=0)
    mean[cls] = mean[cls].reshape((-1, 1))
    i = i+1

# compute covariance matrix
s = {}
normalized_features = {}
for cls in unique:
    indices = np.flatnonzero(C == cls)
    normalized_features[cls] = X[indices].T - mean[cls]

j = 0
for cls in unique:
    s[cls] = (1/examples_per_class[j]) * (normalized_features[cls] @ normalized_features[cls].T)
    j = j+1

covariance = (examples_per_class[0]/nr_training_examples) * s[unique[0]] + \
             (examples_per_class[1]/nr_training_examples) * s[unique[1]]

cov_inverse = np.linalg.pinv(covariance)
# compute posterior probability
sigmoid = lambda a: 1 / (1 + np.exp(-a))


plt.figure()
plt.xticks(np.arange(0, 19, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel('accuracy')
plt.xlabel('features')
plt.title('Probabilistic generative model accuracy vs number of features')
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


def classify(X, mean, covariance, target, flag="train"):
    result = np.zeros((X.shape[1]-1, 2), dtype=float)
    for features in range(2, X.shape[1]+1):
        sub_m1 = mean[2][0:features]
        sub_m2 = mean[4][0:features]
        covinverse = np.linalg.pinv(covariance[0:features, 0:features])
        weights = covinverse @ (sub_m1-sub_m2)
        bias = (-(1/2) * sub_m1.reshape((1, -1)) @ covinverse @ sub_m1) \
               + ((1/2) * sub_m2.reshape((1, -1)) @ covinverse @ sub_m2) \
               + np.log(priors[2]/priors[4])
        input = X[:, 0:features]
        probability = (sigmoid(weights[0:features].reshape((1, -1)) @ input.T + bias))

        probability[probability > 0.5] = unique[0]
        probability[probability <= 0.5] = unique[1]

        comparison = np.equal(probability.reshape((-1,1)), target)
        accuracy = np.count_nonzero(comparison)/target.size
        print(flag, "---Nr of features ", features, " accuracy", accuracy)

        result[features-2][0] = features
        result[features-2][1] = accuracy

    plt.plot(result[:, 0], result[:, 1], '+-', label=flag)
    plot_point_values(result, plt.gca(), flag)


classify(X, mean, covariance, C)
classify(Xtst, mean, covariance, Ctst, flag="test")

ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.show()

