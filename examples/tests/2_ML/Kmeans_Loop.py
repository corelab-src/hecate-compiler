import hecate as hc
from random import *
import sys

from pathlib import Path
import time

import numpy as np
from sklearn.datasets import make_blobs
import hecate.parser as UTIL

argv = UTIL.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, epochs, input_data = argv

a_compile_type = compile_type
a_compile_opt = int(waterline)
a_epoch = int(epochs)
hc.setLibnHW(argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
hevm.load(
    f"traced/_hecate_{stem}.cst",
    f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm",
)


# argmin - need to compare values to assign data to clusters
# 2 clusters is fine, but more than two clusters can cause problems in HE Kmeans
def assign_clusters(X, centroids):
    distances = np.array(
        [np.sum((X - centroid) ** 2, axis=1) for centroid in centroids]
    )
    # print(distances[:16])

    # data is assigned to the closer centroid
    # print(np.argmin(distances, axis=0)[:16])
    return np.argmin(distances, axis=0)


# need mask for each cluster
def compute_centroids(X, clusters, n_clusters):
    return np.array([X[clusters == k].mean(axis=0) for k in range(n_clusters)])


def Kmeans(X, n_clusters=2, max_iter=10, tol=1e-4):
    # Initialize centroids
    selected_indices = [0, 1]
    centroids = X[selected_indices]

    for i in range(max_iter):
        # Assign clusters
        clusters = assign_clusters(X, centroids)

        # Compute new centroids
        new_centroids = compute_centroids(X, clusters, n_clusters)
        # print(f"Iteration {i+1}, new centroids:\n{new_centroids}")

        centroids = new_centroids
        # print(centroids)
    return centroids, clusters


# Generate some sample data
X, Y = make_blobs(n_samples=256, centers=2, cluster_std=0.80, random_state=0)

# Run KMeans
epochs = a_epoch
centroids, clusters = Kmeans(X, n_clusters=2, max_iter=epochs)

x = X.flatten()
y = Y.flatten()
hevm.setInput(0, x)
hevm.setInput(1, y)
hevm.setEpoch(0, a_epoch)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() - timer
res = hevm.getOutput()
# res_ = np.concatenate((res[0][:2], res[1][:2]), axis=0)
# cen_ = np.concatenate((centroids[0][:2], centroids[1][:2]), axis=0)

# rms = np.sqrt(np.mean(np.power((res_[:4] - cen_), 2)))
rms = np.sqrt(
    np.mean(
        np.power(res[0][:2] - centroids[0], 2) + np.power(res[1][:2] - centroids[1], 2)
    )
)

# hevm.printer(timer/pow(10,9), rms)
hevm.printer(timer / pow(10, 9), rms, epochs)
