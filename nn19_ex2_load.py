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

# extract examples of class SAAB (2) and VAN (4)
indices = np.flatnonzero((C == 4) | (C == 2))
C = C[indices]
X = X[indices]

indices_tst = np.flatnonzero((Ctst == 4) | (Ctst == 2))
Ctst = Ctst[indices_tst]
Xtst = Xtst[indices_tst]

#########################################
#
# Probabilistic generative model classification
#
#########################################

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
sigmoid = lambda a: np.where(a >= 0, 1 / (1 + np.exp(-a)), np.exp(a) / (1 + np.exp(a))) #numerically stable


def plot_point_values(x_y_array, ax, label):
    for row in x_y_array:
        valign = 'bottom'
        offset_x = row[0]-0.05
        offset = row[1]-0.02
        if label == 'train':
            valign = 'top'
            offset_x = row[0]+0.05
            offset = row[1]+0.02
        ax.annotate('{:.2%}'.format(row[1]), xy=(row[0], row[1]), xytext=(offset_x, offset),
                    ha='left', va=valign)


def classify(X, mean, covariance, target, flag="train"):
    result = np.zeros((X.shape[1]-1, 2), dtype=float)
    for features in range(2, X.shape[1]+1):
        sub_m1 = mean[2][0:features]
        sub_m2 = mean[4][0:features]
        covinverse = np.linalg.pinv(covariance[0:features, 0:features]) # Simga^-1
        weights = covinverse @ (sub_m1-sub_m2) # w
        bias = (-(1/2) * sub_m1.reshape((1, -1)) @ covinverse @ sub_m1) \
               + ((1/2) * sub_m2.reshape((1, -1)) @ covinverse @ sub_m2) \
               + np.log(priors[2]/priors[4]) # w_0
        input = X[:, 0:features]
        prediction = (sigmoid(weights[0:features].reshape((1, -1)) @ input.T + bias)) # p(C_1|x)

        prediction[prediction > 0.5] = unique[0]
        prediction[prediction <= 0.5] = unique[1]

        comparison = np.equal(prediction.reshape((-1,1)), target)
        accuracy = np.count_nonzero(comparison)/target.size
        print(flag, "---Nr of features ", features, " accuracy", accuracy)

        result[features-2][0] = features
        result[features-2][1] = accuracy

    return result


######
# PLOT
######
plt.figure()
plt.xticks(np.arange(0, 19, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel('Classification Accuracy')
plt.xlabel('Features')
plt.title('Probabilistic Generative Model Accuracy VS. Number of Features')
plt.ylim([0, 1])

result = classify(X, mean, covariance, C)
plt.plot(result[:, 0], result[:, 1], '+-', label='train')
plot_point_values(result, plt.gca(), 'train')

result = classify(Xtst, mean, covariance, Ctst, flag="test")
plt.plot(result[:, 0], result[:, 1], '+-', label='test')
plot_point_values(result, plt.gca(), 'test')


ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.show()

#########################################
#
# IRLS classification
#
#########################################

def nabla_e(X, y, target):
    return X @ (y - target)


def hessian(X, y):
    R = np.diag(y[0]*(1-y[0]))
    return X.T @ R @ X


def update(weights, hessian, gradient):
    return weights - (np.linalg.inv(hessian) @ gradient)


def log_likelihood(estimate, target):
    err_per_example = np.where(target != 0, target * np.log(estimate), 0)
    err_per_example = np.where((1-target) != 0, err_per_example + (1-target) * np.log(1 - estimate), 0)
    return -np.sum(err_per_example, axis=0)


def irls_single(input, target):
    weights = np.random.uniform(low=-0.0001, high=0.0001, size=input.shape[1])[:, np.newaxis]
    y = sigmoid(weights.reshape((1, -1)) @ input.T)  # probability of class 1

    error_old = 100
    error_new = 10

    while np.abs(error_old - error_new) > 1:
        error_old = error_new

        # update until error converges
        hess = hessian(input, y)
        gradient = nabla_e(input.T, y.reshape((-1, 1)), target)
        weights = update(weights, hess, gradient)

        y = sigmoid(weights.reshape((1, -1)) @ input.T)
        error_new = log_likelihood(y.reshape((-1, 1)), target)
        print("error:", error_new)

    prediction = np.where(y > 0.5, 1, 0)  # TODO recheck!! This doesn't make sense shouldn't it be classifying class 1, i.e. t = 0??
    comparison = np.equal(prediction.reshape((-1, 1)), target)
    accuracy = np.count_nonzero(comparison)/target.size
    print("accuracy", accuracy)

    return accuracy, weights


def classify_irls(weights, X, target):
    y_tst = sigmoid(weights.reshape((1, -1)) @ X.T)
    prediction = np.where(y_tst > 0.5, 1, 0)
    comparison = np.equal(prediction.reshape((-1, 1)), target)
    accuracy = np.count_nonzero(comparison)/target.size
    print("tst accuracy", accuracy)
    return accuracy


def irls(X, target, Xtest, Ctest, flag="train"):
    result = np.zeros((X.shape[1]-2, 3), dtype=float)
    bias = np.ones((X.shape[0], 1))
    bias_tst = np.ones((Xtest.shape[0], 1))
    for features in range(2, X.shape[1]):
        input = np.append(bias, X[:, 0:features], axis=1)
        input_tst = np.append(bias_tst, Xtest[:, 0:features], axis=1)

        print(flag, "---Nr of features ", features)
        accuracy, weights = irls_single(input, target)
        tst_accuracy = classify_irls(weights, input_tst, Ctest)
        result[features-2][0] = features
        result[features-2][1] = accuracy
        result[features-2][2] = tst_accuracy
    return result


######
# PLOT
######
plt.figure()
plt.xticks(np.arange(0, 19, 1))
plt.yticks(np.arange(0, 1.1, 0.1))
plt.ylabel('Classification Accuracy')
plt.xlabel('Features')
plt.title('IRLS Accuracy VS Number of Features')
plt.ylim([0, 1.1])

result = irls(X, np.where(C == 2, 0, 1), Xtst, np.where(Ctst == 2, 0, 1))
# plot graph
plt.plot(result[:, 0], result[:, 1], '+-', label='train')
plot_point_values(result, plt.gca(), 'train')

plt.plot(result[:, 0], result[:, 2], '+-', label='test')
plot_point_values(np.delete(result, 1, 1), plt.gca(), 'test')  # delete second column containing the train accuracy


ax = plt.gca()
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
plt.show()


#########################################
#
# Decision boundary
#
#########################################


X_2_feat = X[:, :2]
x_axis = X_2_feat[:, 0]
y_axis = X_2_feat[:, 1]

bias = np.ones((X_2_feat.shape[0], 1))
input = np.append(bias, X_2_feat, axis=1)

result, weights = irls_single(input, np.where(C == 2, 0, 1))

x = - weights[0] / weights[1]
y = - weights[0] / weights[2]
k = - y / x

line = k * x + y

plt.figure()
ax = plt.gca()
colors = np.squeeze(C)
scatter = plt.scatter(X_2_feat[:, 0], X_2_feat[:, 1], marker=(5, 1), c=colors)
#plt.plot([0, x], [y, line], '-r')
plt.xlabel('x1')
plt.ylabel('x2')
legend1 = ax.legend(*scatter.legend_elements(),
                    loc="upper left", title="Classes")
ax.add_artist(legend1)
plt.title('decision boundaries')

plt.show()

