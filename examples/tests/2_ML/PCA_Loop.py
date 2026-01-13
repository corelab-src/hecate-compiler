import hecate as hc
from random import *
import numpy as np
import sys
from pathlib import Path
import time
import hecate.parser as UTIL

argv = UTIL.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv
seed(100)
a_compile_type = compile_type
a_compile_opt = int(waterline)
a_epoch = int(loop_count)
sqrt_epoch = int(loop_count)
hc.setLibnHW(argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load(
    f"traced/cst/_hecate_{stem}.cst",
    f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm",
)

from sklearn import datasets

iris = datasets.load_iris()
X, Y = iris.data, iris.target
n_samples, n_features = X.shape
n_components = 4

# covariance, eigenvalue, eigenvector
X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
cov = np.dot(X.T, X) / (n_samples - 1)

x = X.flatten()
y = Y.flatten()

epochs = a_epoch
epochs_power = epochs
epochs_babyl = sqrt_epoch
epochs_newton = 4
# epochs_babyl = math.ceil(a_epochs / 5)


def power_iteration(cov, epoch):
    b_k = np.ones(cov.shape[1])
    for _ in range(epoch + 1):
        b_k1 = np.dot(cov, b_k)
        b_k1_squared = np.dot(b_k1, b_k1)
        b_k1_norm = sqrt_babylonian(b_k1_squared, epochs_babyl)
        b_k1_norm_i = newton_raphson_inverse(b_k1_norm, epochs_newton)
        b_k = b_k1 * b_k1_norm_i
    return b_k


def sqrt_babylonian(n, epoch):
    x = n * 0.5
    for _ in range(epoch):
        x_i = newton_raphson_inverse(x, epochs_newton)
        x = 0.5 * (x + n * x_i)
    return x


def newton_raphson_inverse(n, epoch):
    x = 0.1
    for _ in range(epoch):
        x = x * (2 - n * x)
    return x


eigenvectors = []
for i in range(n_components):
    eigenvector = power_iteration(cov, epochs_power)

    # Finding the largest eigenvaluei
    inv = np.dot(eigenvector.T, eigenvector)
    inv = newton_raphson_inverse(inv, epochs_newton)

    eigenvalue = np.dot(eigenvector.T, np.dot(cov, eigenvector)) * inv

    # Deflating cov to obtain cov_new
    cov -= eigenvalue * np.outer(eigenvector, eigenvector)

    eigenvectors.append(eigenvector)

hevm.setInput(0, x)
hevm.setInput(1, y)
hevm.setEpoch(0, a_epoch)
hevm.setEpoch(1, sqrt_epoch)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() - timer
res = hevm.getOutput()

rms = np.sqrt(
    np.mean([np.power(abs(res[0][i] - eigenvectors[0][i]), 2) for i in range(4)])
)
hevm.printer(timer / pow(10, 9), rms, epochs)
print("epoch2:", sqrt_epoch)
