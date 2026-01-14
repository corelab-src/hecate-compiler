import hecate as hc
from random import *
import numpy as np
import sys
from pathlib import Path
import time

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv
seed(100)
a_compile_type = compile_type
a_compile_opt = int(waterline)
a_epoch = int(loop_count)
hc.setLibnHW(argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load(
    f"traced/cst/_hecate_{stem}.cst",
    f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm",
)

from sklearn import datasets

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
# hevm.setEpoch(0, a_epoch)
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
# rms = math.sqrt(np.power(res[0][0] - b[0][0][0], 2))
# for i in range(1,31):
#     rms += math.sqrt(np.power(res[i][0] - W[i-1][0][0], 2))


# print (timer / pow(10,9))
# print(rms)
hevm.printer(timer / pow(10, 9), rms, epochs)
# hevm.printer(timer/pow(10,9), rms)
