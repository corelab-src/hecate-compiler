import hecate as hc
from random import *
import numpy as np
from sklearn import datasets
import sys
from pathlib import Path
import time

a_compile_type = sys.argv[1]
a_compile_opt = int(sys.argv[2])
a_epoch = int(sys.argv[5])
hc.setLibnHW(sys.argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load(
    f"traced/_hecate_{stem}.cst",
    f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm",
)

### unpacked version ###
# bc = datasets.load_breast_cancer()
# X, Y = bc.data, bc.target
# n_samples, n_features = X.shape
# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

# XT= X.T
# XT = np.pad(XT, ((0, 0), (0, 1024 - XT.shape[1])), 'constant')
# Y = np.pad(Y, (0, 1024 - Y.shape[0]), 'constant')
# Y = Y.astype(np.float64)
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))

# def sum_elements(array, size):
#     for i in range(10):
#         rot = np.roll(array, -2**i)
#         array = array + rot
#     return array

# mask = np.concatenate((np.ones(n_samples), np.zeros(1024-n_samples)))

# # W = [np.ones(1024) for _ in range(n_features)]
# # b = [np.ones(1024) for _ in range(1)]
# # W = [np.zeros(1024) for _ in range(n_features)]
# # b = [np.zeros(1024) for _ in range(1)]
# W = [[np.full(1024, 0.0000001)] for _ in range(n_features)]
# b = [[np.full(1024, 0.0000001)] for _ in range(1)]

# learning_rate = -0.0001
# learning_rate_avg = -0.0001/n_samples

# ### epochs ###
# epochs = a_epoch

# for _ in range(epochs):
#     linear_model = [np.zeros(1024)]
#     for i in range(n_features):
#         linear_model += XT[i] * W[i]
#     linear_model += b[0]

#     y_predict = sigmoid(linear_model)

#     error = [np.zeros(1024) for _ in range(n_features)]
#     gradW = [np.zeros(1024) for _ in range(n_features)]
#     gradb = [np.zeros(1024) for _ in range(1)]

#     # calculate weight gradient
#     for i in range(n_features):
#         error[i] = (y_predict - Y) * XT[i]
#         gradW[i] = sum_elements(error[i], 1024)
#         gradW[i] = gradW[i] * learning_rate_avg

#     # calculate bias gradient
#     gradb[0] = (y_predict - Y) * mask
#     gradb[0] = sum_elements(gradb[0], 1024)
#     gradb[0] = gradb[0] * learning_rate_avg

#     # apply weight and bias gradient
#     for i in range(n_features):
#         W[i] += gradW[i]
#     b[0] += gradb[0]

# Call the HE function
# for i in range(30):
#     hevm.setInput(i, np.copy(XT[i]))
# hevm.setInput(30, Y)
# hevm.setEpoch(0, a_epoch)
# timer = time.perf_counter_ns()
# hevm.run()
# timer = time.perf_counter_ns() -timer
# res = hevm.getOutput()
# print("XT: ", XT.shape, XT[0][:3])
# print("True", "\nbias: ", b[0][0][0], "\nWeights: ", W[0][0][0], W[1][0][0], W[2][0][0], W[3][0][0])
# print("HE",   "\nbias: ", res[0][0],  "\nWeights: ", res[1][0], res[2][0], res[3][0], res[4][0])
# calculate root mean square
# import math
# rms = math.sqrt(np.power(res[0][0] - b[0][0][0], 2))
# for i in range(1,31):
#     rms += math.sqrt(np.power(res[i][0] - W[i-1][0][0], 2))

# print (timer / pow(10,9))
# print(rms)

### packed version ###
bc = datasets.load_breast_cancer()
X, Y = bc.data, bc.target
n_samples, n_features = X.shape
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
x_data = [X.T[i] for i in range(n_features)]
pad = [0.0 for _ in range(2048 - n_samples)]
x = [1.0 for _ in range(n_samples)] + pad
y = Y.tolist() + pad
for samples in x_data:
    samples = samples.tolist()
    x += samples + pad

W = np.zeros(n_features)
b = 0.0
learning_rate = -0.0001
epochs = a_epoch


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


for i in range(epochs):
    linear_model = np.dot(X, W) + b
    y_predicted = _sigmoid(linear_model)

    # compute gradients
    dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y))
    db = (1 / n_samples) * np.sum(y_predicted - Y)

    # update parameters
    W += learning_rate * dw
    b += learning_rate * db

hevm.setInput(0, x)
hevm.setInput(1, y)
hevm.setEpoch(0, a_epoch)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() - timer
res = hevm.getOutput()
# resW = [res[i][0] for i in range(31)]
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2) + np.power(res[1] - c, 2)))
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2)[:4096] + np.power(res[1] - c, 2)[:4096]))
import math

rms = math.sqrt(np.power(res[0][0] - b, 2))
for i in range(1, 31):
    rms += math.sqrt(np.power(W[i - 1] - res[i][0], 2))

hevm.printer(timer / pow(10, 9), rms, epochs)
# hevm.printer(timer/pow(10,9), rms)
