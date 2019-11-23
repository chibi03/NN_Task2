import pickle as pckl  # to load dataset
import pylab as pl     # for graphics
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
file_in = open('vehicle.pkl', 'rb')
vehicle_data = pckl.load(file_in)
file_in.close()


tmp_X = vehicle_data['train']['X']  # features; X[i,j]...feature j of example i
tmp_C = vehicle_data['train']['C']  # classes; C[i]...class of example i

indices = np.flatnonzero(tmp_C == 2)
C_1 = tmp_C[indices]
X_1 = tmp_X[indices]

indices = np.flatnonzero(tmp_C == 4)
C_2 = tmp_C[indices]
X_2 = tmp_X[indices]

plt.figure()
plt.title('Probabilistic Generative Model Decision Boundary for 2 Features')
plt.scatter(X_1[:,0:1], X_1[:,1:2])
plt.scatter(X_2[:,0:1], X_2[:,1:2])
plt.xlabel('Compactness')
plt.ylabel('Circularity')
plt.show()