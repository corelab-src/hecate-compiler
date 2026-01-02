
import hecate as hc
from random import *
import numpy as np
import sys
from pathlib import Path
import time

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, epochs, input_data = argv

seed(100)
a_compile_type = compile_type
a_compile_opt = int(waterline)
a_epoch = int(epochs)
hc.setLibnHW(argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

from sklearn import datasets

X, y = datasets.make_blobs(n_samples=1024, n_features=2, centers=2, cluster_std=1, random_state=0)
X0 = [item[0] for item in X]
X1 = [item[1] for item in X]
Y = np.where(y == 0, -1, 1)

lambda_param = 0.01
learning_rate = -0.001
lr_lambda = lambda_param * learning_rate

n_samples, n_features = X.shape
W0 = np.ones(n_samples)
W1 = np.ones(n_samples)
b0 = np.ones(n_samples)
# W = np.zeros(2)
# b = 0.0
cond = []

epochs = a_epoch 

def sum_elements(array, size):
    for i in range(int(np.log2(size))):
        rot = np.roll(array, -2**i)
        array = array + rot
    return array

# Training process
for _ in range(epochs):
    func = X0 * W0 + X1 * W1 - b0
    func = Y * func - np.ones(n_samples)

    # label the result based on function
    y_predict = np.sign(func)

    cond0 =  0.5 * y_predict + 0.5    # correct classification
    cond1 = -0.5 * y_predict + 0.5    # wrong classification

    # correct classification weights are updated to only include the regularization term
    gradW0 = cond0 * (2 * lambda_param * W0)
    gradW1 = cond0 * (2 * lambda_param * W1)

    gradW0 += cond1 * (2 * lambda_param * W0 - X0 * Y)
    gradW1 += cond1 * (2 * lambda_param * W1 - X1 * Y)
    gradb0 = cond1 * Y

    # sum the gradients
    gradW0 = sum_elements(gradW0, n_samples)
    gradW1 = sum_elements(gradW1, n_samples)
    gradb0 = sum_elements(gradb0, n_samples)

    # apply the gradients
    W0 = W0 + gradW0 * learning_rate
    W1 = W1 + gradW1 * learning_rate
    b0 = b0 + gradb0 * learning_rate

# for i in range(epochs):
#     w_update_sum = []
#     b_update_sum = []
#     for idx, x_i in enumerate(X):
#         condition = y[idx] * (np.dot(x_i, W) - b) >= 1
#         # print(idx, x_i, W, b)
#         # cond.append(condition)
#         if condition:
#             w_update_sum.append(lambda_param * W)
#         else:                                           
#             w_update_sum.append(lambda_param * W - x_i *  y[idx])
#             b_update_sum = np.append(b_update_sum, y[idx])
            
#     # print(w_update_sum[:10], b_update_sum[:10])
            
#     w_update_sum = np.array(w_update_sum)
#     b_update_sum = np.array(b_update_sum)

#     w_update_sum = np.sum(w_update_sum, axis=0)
#     b_update_sum = np.sum(b_update_sum)

#     W += w_update_sum * learning_rate
#     b += b_update_sum * learning_rate

Y = Y.astype(np.float64)
hevm.setInput(0, X0)
hevm.setInput(1, X1)
hevm.setInput(2, Y)
hevm.setEpoch(0, a_epoch)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
res = hevm.getOutput()
import math
# rms = math.sqrt(np.power(res[0][0] - W[0], 2) + np.power(res[1][0] - W[1], 2) + np.power(res[2][0] - b, 2))

se  = np.power(res[0][0] - W0[0], 2) + np.power(res[1][0] - W1[0], 2) + np.power(res[2][0] - b0[0], 2)
mse = np.mean(se)
rms = math.sqrt(mse)

# hevm.printer(timer/pow(10,9), rms)
hevm.printer(timer/pow(10,9), rms, epochs)
