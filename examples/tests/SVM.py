
import hecate as hc
from random import *
import numpy as np
import sys
from pathlib import Path
import time

seed(100)
a_compile_type = sys.argv[1]
a_compile_opt = int(sys.argv[2])
hc.setLibnHW(sys.argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load (f"traced/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")

from sklearn import datasets

bc = datasets.load_breast_cancer()
X, y = bc.data, bc.target
n_samples, n_features = X.shape

X, y = datasets.make_blobs(n_samples=4096, n_features=2, centers=2, cluster_std=1, random_state=0)
x1 = [item[0] for item in X]
x2 = [item[1] for item in X]
x = x1 + x2
y = np.where(y == 0, -1, 1)

lambda_param = 0.02
W = np.zeros(2)
b = 0.0
cond = []

learning_rate = -0.0001
epochs = 10 

for i in range(epochs):
    w_update_sum = []
    b_update_sum = []
    for idx, x_i in enumerate(X):
        condition = y[idx] * (np.dot(x_i, W) - b) >= 1
        # print(idx, x_i, W, b)
        # cond.append(condition)
        if condition:
            w_update_sum.append(lambda_param * W)
        else:                                           
            w_update_sum.append(lambda_param * W - x_i *  y[idx])
            b_update_sum = np.append(b_update_sum, y[idx])
            
    # print(w_update_sum[:10], b_update_sum[:10])
            
    w_update_sum = np.array(w_update_sum)
    b_update_sum = np.array(b_update_sum)

    w_update_sum = np.sum(w_update_sum, axis=0)
    b_update_sum = np.sum(b_update_sum)

    W += w_update_sum * learning_rate
    b += b_update_sum * learning_rate


hevm.setInput(0, x)
hevm.setInput(1, y)
hevm.run()
hevm.setInput(0, x)
hevm.setInput(1, y)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
res = hevm.getOutput()
print(res.shape)
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2) + np.power(res[1] - c, 2)))
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2)[:4096] + np.power(res[1] - c, 2)[:4096]))
import math
rms = math.sqrt(np.power(res[0][0] - W[0], 2)[:4096] + np.power(res[1][0] - W[1], 2)[:4096] + np.power(res[2][0] - b, 2)[:4096])

# print (timer / pow(10,9))
# print(rms)
hevm.printer(timer/pow(10,9), rms)
