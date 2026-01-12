#!/usr/bin/env python

import faulthandler

faulthandler.enable()

import sys
import os
import json
from pathlib import Path
import hecate as hc
import poly.MPCB as MPCB
import poly.Func as Func
from poly.Poly import *


# ------------------------------------------------------------------
# Global Configurations & Constants
# ------------------------------------------------------------------
def bootstrap(out):
    if out is list:
        if len(out) == 1:
            out = hc.bootstrap(out)
        for i in out:
            out[i] = hc.bootstrap(out[i])
    else:
        out = hc.bootstrap(out)
    return out


source_path = Path(__file__).resolve()
source_dir = source_path.parent
hecate_dir = os.environ["HECATE"]
# coeff_dir = hecate_dir + "/python/poly/poly/data/"
coeff_dir = hecate_dir + "/../hera/legendre/"
# Need to modify files
# inv_coeff_file = hecate + folder + "inv_legendre_coeffs_31_9.0e-01_1.1e+00.txt"
inv_coeff_file = coeff_dir + "inv_legendre_coeffs_3_9.0e-01_1.1e+00.txt"
max_inv, min_inv = 1.1, 0.9

# sqrt_inv_coeff_file = coeff_dir + "inv_sqrt_legendre_coeffs_7_1.0e+00_4.0e+01.txt"
sqrt_inv_coeff_file = coeff_dir + "inv_sqrt_legendre_coeffs_7_7.0e+00_2.1e+01.txt"
max_inv_sqrt, min_inv_sqrt = 21, 7


def read_coeffs(filepath):
    with open(filepath, "r") as file:
        return [float(line.strip()) for line in file if line.strip()]


inv_coeffs = read_coeffs(inv_coeff_file)
sqrt_inv_coeffs = read_coeffs(sqrt_inv_coeff_file)
Inv_approx = lambda x, Xmin, Xmax: _eval_legendre(x, inv_coeffs, Xmin, Xmax)
Sqrt_Inv_approx = lambda x, Xmin, Xmax: _eval_legendre(x, sqrt_inv_coeffs, Xmin, Xmax)

argv = hc.hc_parser(__file__)
compile_opt, waterline, benchmark, library, hardware, epochs, input_data = argv
config_name = f"profiled_{library}_{hardware}.json"
with open(str(hecate_dir) + "/" + config_name, "r") as f:
    run_config = json.load(f)
    config_N = run_config["polynomialDegree"]
    config_L = run_config["levelUpperBound"]

slot_length = config_N >> 1
N = slot_length
print(f"slot_length: {slot_length}")
NUM_ROWS, NUM_COLS = 150, 4


# ------------------------------------------------------------------
# Math & Logic Helper Functions
# ------------------------------------------------------------------
def sum_rot(data, steps, shift):
    for i in range(steps):
        data += data.rotate(shift * (1 << i))
    return data


sum_reduce_stride_4 = lambda data, steps: sum_rot(data, steps, 4)
sum_reduce_stride_1 = lambda data, steps: sum_rot(data, steps, 1)


def generate_pca_masks(columns, rows):
    mask_list = [[] for _ in range(6)]
    mask_list[0] = [1.0] * 4 + [0.0] * (1024 - 4)
    mask_list[1] = [1.0] * 600 + [0.0] * (1024 - 600)
    mask_list[2] = [1.0] + [0.0] * (1024 - 1)
    mask_list[3] = [1.0, 0.0, 0.0, 0.0]
    mask_list[4] = [1.0, 0.0, 0.0, 0.0] * 4 + [0.0] * (1024 - 16)
    mask_list[5] = [1.0] * 16 + [0.0] * (1024 - 16)
    return mask_list


def create_bit_mask(n):
    # temp[1] = [0.0, 1.0, 0.0, 0.0 ...]
    return [[1.0 if j == i else 0.0 for j in range(n)] for i in range(4)]


def replicate_slots(x, mask, n):
    x = x * mask
    x = x + x.rotate(1 * n) + x.rotate(2 * n) + x.rotate(3 * n)
    x = x.rotate(1024 - 3 * n)
    return x


# ------------------------------------------------------------------
# Legendre Method (Inverse, Sqrt)
# ------------------------------------------------------------------
# Xmin, Xmax = 1, 40
def _eval_legendre(x, coeffs, Xmin, Xmax):
    # hc.set_range(x, [Xmin, Xmax])
    center = Xmin + Xmax
    scale = Xmax - Xmin
    # 1) Normalize x to [-1, 1]
    x = (2.0 * x + (-center)) * (1.0 / scale)

    # 2) Number of polynomials we need
    n = len(coeffs) - 1  # If we have N+1 coefficients, the highest P is P_N

    # 3) Prepare an array P to hold P_k(x)
    P = [None] * (n + 1)

    # --- P0(x) = 1
    P[0] = 1.0  # always needed

    # --- P1(x) = x
    if n >= 1:
        P[1] = x

    # --- P2(x) explicitly if n >= 2
    if n >= 2:
        # Using direct formula for P2(x) = (3/2)x^2 - 1/2
        P[2] = ((3.0 * (x * x)) + (-1.0)) * (1.0 / 2.0)

    # --- Build P3, P4, ..., Pn if n > 2
    # Using the factor-out form:
    #   P_{k+1}(x) = 1/(k+1) * [ (2k+1) x P_k(x) - k P_{k-1}(x) ]
    for k in range(2, n):
        P[k + 1] = ((2.0 * k / (k + 1) + 1.0 / (k + 1)) * x * P[k]) - (
            (k / (k + 1)) * P[k - 1]
        )

    # 4) Sum up coefficient_k * P_k(x)
    result = [coeffs[0]]
    # Handle k = 1 (P1)
    if n >= 1:
        result = result + (coeffs[1] * P[1])

    # Now handle k >= 2
    for k in range(2, len(coeffs)):
        result = result + (coeffs[k] * P[k])

    return result


# ------------------------------------------------------------------
# PCA Core Logic
# ------------------------------------------------------------------


def compute_covariance_matrix(x_data, columns, rows, bit_mask):
    cov = 0.0
    for i in range(1):
        temp = x_data * x_data
        temp = sum_reduce_stride_4(temp, 8)  # Reduce columns
        for k in range(4 - i):
            num = temp * bit_mask[k]
            cov += num.rotate(16 - 4 * k - i)

    # Off-diagonal elements logic
    for i in range(1, 4):
        temp = x_data * x_data.rotate(i)
        temp = sum_reduce_stride_4(temp, 8)
        for k in range(4 - i):
            num = temp * bit_mask[k]
            # Accumulate into Cov matrix structure
            cov += num.rotate(16 - 4 * k - i)
            cov += num.rotate(16 - 4 * k - 4 * i)

    # Normalize by (Rows - 1)
    cov = cov.rotate(1024 - 16)
    cov = cov * (1 / (rows - 1.0))
    return cov


def vector_matrix_dot(cov, x, mask):
    x = replicate_slots(x, mask[0], 4)
    x = cov * x
    x = sum_reduce_stride_1(x, 2)
    x = x * mask[3]
    x = x + x.rotate(3) + x.rotate(6) + x.rotate(9)
    return x


def power_iteration(cov, epochs, mask):
    b_k = mask[0]
    cov_sum = sum_reduce_stride_4(cov, 2)

    # First iteration setup
    b_k1 = b_k * cov_sum
    b_k1_squared = b_k1 * b_k1
    b_k1_squared = sum_reduce_stride_1(b_k1_squared, 2)

    # Normalize
    b_k1_norm_i = Sqrt_Inv_approx(b_k1_squared, Xmin=min_inv_sqrt, Xmax=max_inv_sqrt)
    b_k1_norm_i = replicate_slots(b_k1_norm_i, mask[2], 1)
    b_k = b_k1 * b_k1_norm_i
    b_k = bootstrap(b_k)

    # Iterative refinement
    for _ in range(epochs):
        # b_{k+1} = Cov * b_k
        b_k1_in = vector_matrix_dot(cov, b_k, mask)

        # Normalize: b_{k+1} / ||b_{k+1}||
        b_k1_squared_in = b_k1_in * b_k1_in
        b_k1_squared_in = sum_reduce_stride_1(b_k1_squared_in, 2)

        # b_k1_squared_in = bootstrap(b_k1_squared_in)
        b_k1_norm_i_in = Sqrt_Inv_approx(
            b_k1_squared_in, Xmin=min_inv_sqrt, Xmax=max_inv_sqrt
        )
        b_k1_norm_i_in = replicate_slots(b_k1_norm_i_in, mask[2], 1)
        b_k1_norm_i_in = bootstrap(b_k1_norm_i_in)
        b_k = b_k1_in * b_k1_norm_i_in
        # b_k = bootstrap(b_k)

    return b_k


# ------------------------------------------------------------------
# Hecate Trace Function
# ------------------------------------------------------------------
@hc.func("c,c")
def PCA(x_data, y_data):
    mask = generate_pca_masks(NUM_COLS, NUM_ROWS)
    bit_mask = create_bit_mask(1024)
    x_data = x_data * mask[1]
    cov = compute_covariance_matrix(x_data, NUM_COLS, NUM_ROWS, bit_mask)

    num_components = 1
    for i in range(num_components):

        # A. Find dominant eigenvector using Power Iteration
        cov = bootstrap(cov)
        eigenvector = power_iteration(cov, epochs, mask)

        # B. Calculate Eigenvalue: (v^T * Cov * v) / (v^T * v)
        # Assuming eigenvector is normalized, denominator is ~1, but calculating explicitly
        dot1 = vector_matrix_dot(cov, eigenvector, mask)  # Cov * v
        dot2 = eigenvector * dot1  # v * (Cov * v)
        dot2 = sum_reduce_stride_1(dot2, 2)  # Sum to get scalar

        inv1 = eigenvector * eigenvector  # v * v
        inv1 = sum_reduce_stride_1(inv1, 2)  # Norm squared
        inv2 = Inv_approx(inv1, Xmin=0.9, Xmax=1.1)

        eigenvalue = dot2 * inv2

        # C. Deflation (Remove current PC from Covariance Matrix)
        # Cov_new = Cov - eigenvalue * v * v^T

        # Prepare expanded forms for deflation
        copy_eigenvalue = replicate_slots(eigenvalue, mask[2], 1)
        copy_eigenvalue = replicate_slots(copy_eigenvalue, mask[0], 4)
        copy_eigenvector = replicate_slots(eigenvector, mask[0], 4)

        # Align eigenvector for outer product subtraction
        for k in range(1, 4):
            element = eigenvector * bit_mask[k]
            eigenvector += element.rotate(1024 - 3 * k)
            # eigenvector += element.rotate(1024-3*i)

        long_eigenvector = replicate_slots(eigenvector, mask[3], 1)

        # Compute Delta Covariance
        del_cov = copy_eigenvalue * copy_eigenvector * long_eigenvector
        cov = cov - del_cov
        cov = cov * mask[5]  # Clean up matrix region

    return eigenvector


if __name__ == "__main__":
    modName = hc.save("traced", "traced")
    print(f"Module saved: {modName}")
