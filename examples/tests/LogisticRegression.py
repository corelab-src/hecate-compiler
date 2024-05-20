
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

x = [ uniform (-1, 1) for a in range(4096)]
a = 2.0
b = 1.0
y = [ a*point +b + uniform (-0.01, 0.01) for point in x]


# print(res)

epochs = 10
learning_rate = -0.01

W = np.zeros(n_features)
b = 0.0  
learning_rate = -0.0001
epochs = 10 

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

for i in range(epochs):
    linear_model = np.dot(X, W) + bias
    y_predicted = _sigmoid(linear_model)

    # compute gradients
    dw = (1 / n_samples) * np.dot(X.T, (y_predicted - Y))
    db = (1 / n_samples) * np.sum(y_predicted - Y)

    # update parameters
    W += learning_rate * dw
    b += learning_rate * db


hevm.setInput(0, x)
hevm.setInput(1, y)
hevm.run()
hevm.setInput(0, x)
hevm.setInput(1, y)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() -timer
res = hevm.getOutput()
resW = [res[i * 4096] for i in range(31)]
print("TEST RESULT")
print (W, c)
print(res[0], res[1])
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2) + np.power(res[1] - c, 2)))
# rms = np.sqrt(np.mean(np.power(res[0] - W, 2)[:4096] + np.power(res[1] - c, 2)[:4096]))
import math
rms = math.sqrt(pow(resW[0] - b, 2))
for i in range(1,31):
    rms += math.sqrt(pow(W[i-1] - resW[i], 2))


# print (timer / pow(10,9))
# print(rms)
hevm.printer(timer/pow(10,9), rms)
