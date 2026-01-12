#!/usr/bin/env python

import faulthandler

faulthandler.enable()

import torch
import numpy as np
import hecate as hc
import time
import sys
import os
import json
from pathlib import Path
from sklearn import datasets

source_path = Path(__file__).resolve()
source_dir = source_path.parent
hecate_dir = os.environ["HECATE"]

argv = hc.hc_parser(__file__)
compile_opt, waterline, benchmark, library, hardware, epochs, input_data = argv

config_name = f"profiled_{library}_{hardware}.json"
with open(str(hecate_dir) + "/" + config_name, "r") as f:
    run_config = json.load(f)
    config_N = run_config["polynomialDegree"]
    config_L = run_config["levelUpperBound"]

slot_length = config_N >> 1
print(f"slot_length: {slot_length}")


# ------------------------------------------------------------------
# Python Reference Function (Approximation Logic)
# ------------------------------------------------------------------
def python_reference_pca(x, y, cov, n_components, epochs):
    def power_iteration(cov, epochs):
        b_k = np.ones(cov.shape[1])
        for _ in range(epochs + 1):
            b_k1 = np.dot(cov, b_k)
            b_k1_squared = np.dot(b_k1, b_k1)
            b_k1_norm_i = 1 / b_k1_squared
            b_k1_norm = np.sqrt(b_k1_norm_i)
            print(
                f"inv_input: {b_k1_squared}, sqrt_tmp: {b_k1_norm_i}, inv_sqrt_result: {b_k1_norm}"
            )
            b_k = b_k1 * b_k1_norm
        return b_k

    eigenvectors = []

    for i in range(n_components):
        eigenvector = power_iteration(cov, epochs)

        # Finding the largest eigenvaluei
        inv_input = np.dot(eigenvector.T, eigenvector)
        inv = 1 / inv_input
        # inv = newton_raphson_inverse(inv_input, epochs)
        print(f"inv_input: {inv_input}, inv_result: {inv}")
        print(
            f"max(inv_input): {np.max(inv_input)}, min(inv_input): {np.min(inv_input)}"
        )
        print(f"max(inv_result): {np.max(inv)}, min(inv_result): {np.min(inv)}")

        eigenvalue = np.dot(eigenvector.T, np.dot(cov, eigenvector)) * inv

        # Deflating cov to obtain cov_new
        cov -= eigenvalue * np.outer(eigenvector, eigenvector)
        eigenvectors.append(eigenvector)

    return np.array(eigenvectors[0])


def get_ground_truth_pca(X):
    n_samples, n_features = X.shape

    cov = np.dot(X.T, X) / (n_samples - 1)

    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    return eigenvectors[:, 0]


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    iris = datasets.load_iris()
    X, Y = iris.data, iris.target
    n_samples, n_features = X.shape

    # covariance, eigenvalue, eigenvector
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    x = X.flatten()
    y = Y.flatten()
    ref_timer_start = time.perf_counter_ns()
    cov = np.dot(X.T, X) / (n_samples - 1)
    ref_tensor_np = python_reference_pca(x, y, cov, 1, epochs)

    # ref_tensor_np = get_ground_truth_pca(X)
    ref_tensor = torch.from_numpy(ref_tensor_np)
    ref_execution_time = (time.perf_counter_ns() - ref_timer_start) / pow(10, 9)

    mem_before = hc.print_mem("Before hevm.run()")
    hc.setLibnHW(argv)  # Set the path to the libnHW.so
    hevm = hc.HEVM()
    hevm.load(
        f"traced/_hecate_{benchmark}.cst",
        f"optimized/{compile_opt}/{benchmark}.{waterline}._hecate_{benchmark}.hevm",
    )
    hevm.setInput(0, x)
    hevm.setInput(1, y)

    timer_start = time.perf_counter_ns()
    hevm.run()
    execution_time = (time.perf_counter_ns() - timer_start) / pow(10, 9)

    mem_after = hc.print_mem("After hevm.run()")
    mem_diff = mem_after - mem_before

    # 6. Evaluation & Comparison
    print("================================================")
    print("Evaluation result")
    print("================================================")
    res = hevm.getOutput()

    # Extract relevant output (first 4 elements corresponding to the eigenvector)
    # The output from Hecate is full slot_length, we need the first N_Features (4)
    result_np = res[0][:4]
    result_tensor = torch.from_numpy(result_np)

    # --- SIGN CORRECTION ---
    # PCA Eigenvectors have arbitrary sign. v and -v are the same axis.
    # If dot product is negative, flip the HE result for fair comparison.
    dot_prod = torch.dot(result_tensor, ref_tensor)
    if dot_prod < 0:
        print("Note: Sign flip detected. Negating HE result for comparison.")
        result_tensor = -result_tensor

    print(f"{'Ref Result (Truth):':<20} {ref_tensor.numpy()}")
    print(f"{'HE Result (Approx):':<20} {result_tensor.numpy()}")
    rms = np.sqrt(
        np.mean([np.power(abs(result_tensor[i] - ref_tensor[i]), 2) for i in range(4)])
    )
    hevm.printer(execution_time, rms.item())
    print("CPU memory usage: ", mem_diff, "GB")

    # Calculate error percentage (add small epsilon to avoid division by zero)
    diff = (result_tensor - ref_tensor).abs()
    bits = -torch.log2(diff)
    epsilon = 1e-9
    error_percentage = diff / (ref_tensor[:4].abs() + epsilon) * 100

    # # Find the maximum error value position (multi-dimensional support)
    # max_err_idx_flat = torch.argmax(error_percentage)
    # max_err_idx = np.unravel_index(max_err_idx_flat.cpu().numpy(), error_percentage.shape)

    # print("\n================================================")
    # print("Evaluation Result")
    # print("================================================")
    # print(f"HE raw output shape:      {res_he.shape}")
    # print(f"Reference tensor shape:   {ref_tensor.shape}")
    # print(f"Processed result shape:   {result_tensor.shape}")
    # # np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, floatmode='fixed', precision=7)
    # # print(f"res_he: {res_he[0,:128]}")
    # # print(f"Reference tensor:         {ref_tensor}")
    # # print(f"Processed result:         {result_tensor}")
    # print(f"ref (max, min, mean, std): {ref_tensor.max():.7g}, {ref_tensor.min():.7g}, {ref_tensor.mean():.7g}, {ref_tensor.std():.7g}")
    # print(f"res (max, min, mean, std): {result_tensor.max():.7g}, {result_tensor.min():.7g}, {result_tensor.mean():.7g}, {result_tensor.std():.7g}")
    bits_rms = torch.sqrt(torch.mean(bits**2))
    error_rms = torch.sqrt(torch.mean(error_percentage**2))
    print(
        f"bits (max, min, rms, std): {bits.max():.7g}, {bits.min():.7g}, {bits_rms:.7g}, {bits.std():.7g}"
    )
    print(
        f"error% (max, min, rms, std): {error_percentage.max():.7g}, {error_percentage.min():.7g}, {error_rms:.7g}, {error_percentage.std():.7g}"
    )

    # print("================================================")
    # print("Error distribution analysis:")
    # print("================================================")
    # print(f"Total elements: {error_percentage.numel()}")
    # print(f"Errors > 0.1%: {(error_percentage > 0.01).sum().item()}")
    # print(f"Errors > 1%: {(error_percentage > 0.1).sum().item()}")
    # print(f"Errors > 5%: {(error_percentage > 1).sum().item()}")
    # print("================================================")
    # # torch.set_printoptions(precision=4, threshold=float('inf'), linewidth=200)
    # # print(error_percentage)
    # # torch.set_printoptions(profile='default')
    # print(f"{'Max Error percentage:':<25} {torch.max(error_percentage).item():.7g}%")
    # print(f"{'Min Error percentage:':<25} {torch.min(error_percentage).item():.7g}%")
    # print(f"{'Mean Error percentage:':<25} {torch.mean(error_percentage).item():.7g}%")
    # # print("\n------------------------------------------------")
    # # print("Maximum Error Details")
    # # print("------------------------------------------------")
    # # print(f"{'Index:':<18} {list(max_err_idx)}")
    # # print(f"{'HE Result:':<18} {result_tensor[max_err_idx].item():.7g}")
    # # print(f"{'Reference:':<18} {ref_tensor[max_err_idx].item():.7g}")
    # # print(f"{'Difference:':<18} {diff[max_err_idx].item():.7g}")
    # # print(f"{'Error Percentage:':<18} {error_percentage[max_err_idx].item():.7g}%")

    # target = torch.tensor([target])
    # criterion = nn.CrossEntropyLoss()
    # loss_ref = criterion(ref_tensor, target)
    # loss_res = criterion(result_tensor, target)
    # print(f"Loss (ref): {loss_ref.item()}")
    # print(f"Loss (res): {loss_res.item()}")
    # print("\n------------------------------------------------")
    # print("Final Prediction Result")
    # print("------------------------------------------------")
    # pred_ref = torch.argmax(ref_tensor)
    # pred_he = torch.argmax(result_tensor)
    # is_correct_ref = "Correct" if pred_ref.item() == target else "Incorrect"
    # is_correct_he = "Correct" if pred_he.item() == target else "Incorrect"
    # print(f"{'True Label (Target):':<25} {target}")
    # print(f"{'Reference Model Prediction:':<25} {pred_ref.item()} ({is_correct_ref})")
    # print(f"{'HE Model Prediction:':<25} {pred_he.item()} ({is_correct_he})")
    # print("================================================\n")
