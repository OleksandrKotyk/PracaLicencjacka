import math

import matplotlib.pyplot as plt
import numpy as np


def sigmoid(x):
    return [1 / (1 + math.exp(-i)) for i in x]


def relu(x):
    return [max(i, 0) for i in x]


x_axis = np.arange(-10., 10., 0.2)
sig = sigmoid(x_axis)

plt.plot(x_axis, sig)
plt.show()

x_axis = np.arange(-10., 10., 0.2)
sig = relu(x_axis)

plt.plot(x_axis, sig)
plt.show()
