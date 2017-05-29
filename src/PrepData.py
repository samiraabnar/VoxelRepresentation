import csv
from keras.layers import Dense
from keras.models import Sequential
import numpy as np
from scipy import *
from scipy.spatial import *
from functions import *


def binarize_vector(vector, positive_threshold=0.5,negetive_threshold=0.5):

    b_vector = np.zeros_like(np.asarray(vector))
    for i in np.arange(len(b_vector)):
        if vector[i] > 0.7:
            b_vector[i] = 1
        elif vector[i] < -0.7:
            b_vector[i] = -1

    return b_vector

brain_activations_1 = genfromtxt('../data/data.csv', delimiter=',')

activations = brain_activations_1[0]
activations.sort()





from matplotlib import pyplot as plt
plt.style.use('fivethirtyeight')
"""fig, ax = plt.subplots()
ax.plot(np.arange(len(activations)),np.tanh(activations))
ax.plot(np.arange(len(activations)),activations)
plt.show()
"""

bb_activations = []
for b_activation in brain_activations_1:
    the_b_act = binarize_vector(b_activation)
    bb_activations.append(the_b_act)


np.save("binary_brain_activations_1_7",np.asarray(bb_activations))


#plt.imshow(bb_activations, cmap='hot', interpolation='none',aspect='auto')
#plt.show()


number_of_voxels = bb_activations[0].shape[0]
input_dim = bb_activations[0].shape[0]
print("input dim:"+str(input_dim))
input_onehot_vectors = np.eye(input_dim)

inputs = []
outputs = []

for i in np.arange(number_of_voxels):
    print(i)
    for j in np.arange(number_of_voxels):
        if i != j:
            for k in np.arange(len(bb_activations)):
                if bb_activations[k][i] > 0 and bb_activations[k][j] > 0:
                    inputs.append(i)
                    outputs.append(j)

print(len(inputs))
print(len(outputs))

np.save("inputs_7",inputs)
np.save("outputs_7",outputs)

