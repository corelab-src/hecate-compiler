#!/usr/bin/env python
"""
@file Format.py
@brief Format test implementation using Hecate.

This file implements both the reference Python implementation and Hecate implementation
of the format test for comparison and validation purposes. It includes timing and
error measurement functionality.
"""

import hecate as hc

import numpy as np
import time

from random import uniform, seed


def generate_LR_data(input_data, target_slope=2.0, target_intercept=1.0):
    """
    @brief Generate test data for linear regression.

    @param input_data Input data
    @param target_slope Target slope for the linear relationship
    @param target_intercept Target intercept for the linear relationship
    @return Tuple of (x_data, y_data, true_slope, true_intercept)
    """
    seed(100)
    x = input_data
    y = [target_slope * point + target_intercept + uniform(-0.01, 0.01) for point in x]
    return x, y, target_slope, target_intercept


def python_format(x, y, epochs, learning_rate=-0.01):
    length = 4096
    half_length = length // 2
    """
    @brief Reference Python implementation of linear regression.
    
    @param x Input data points
    @param y Target values
    @param epochs Number of training epochs
    @param learning_rate Learning rate for gradient descent
    @return Tuple of (final_weight, final_bias)
    """
    W = 1.0  # initial slope
    c = 0.0  # initial intercept

    for _ in range(epochs):
        error = [W * x[i] + c - y[i] for i in range(length)]
        errX = [error[i] * x[i] for i in range(length)]
        gradW = sum(errX) / half_length
        gradb = sum(error) / half_length
        W = W + learning_rate * gradW
        c = c + learning_rate * gradb

    return W, c


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    """
    @brief Main function for running the format test.

    This function parses command-line arguments, sets up the Hecate environment,
    generates test data, runs the Python reference implementation, and then runs
    the Hecate implementation. It also prints the execution time and error.
    """
    # aft_command : new-test Format --opt dacapo --waterline 40 --library HEAAN --hardware GPU --epochs 2 --input True --padding True

    # Parse command line arguments
    argv = hc.hc_parser(__file__)
    compile_opt, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv
    # eliminate benchmark from argv for hc.setLibnHW

    # Setup Hecate environment
    hc.setLibnHW(argv)  # Set the path to the libnHW.so
    hevm = hc.HEVM()  # Create a HEVM object
    # Load the compiled state trace (cst) and hevm (optimized code) files
    hevm.load(
        f"traced/cst/_hecate_{benchmark}.cst",
        f"optimized/{compile_opt}/{benchmark}.{waterline}._hecate_{benchmark}.hevm",
    )

    # Generate test data
    input_data = [uniform(-1, 1) for a in range(4096)]
    x_data, y_data, true_W, true_c = generate_LR_data(input_data)

    # Run Python reference implementation
    ref_W, ref_c = python_format(x_data, y_data, loop_count)

    # Run Hecate implementation
    # hevm.setInput(0, x_data)
    # hevm.setInput(1, y_data)
    # hevm.setEpoch(0, epochs)
    # hevm.run()

    hevm.setInput(0, x_data)
    hevm.setInput(1, y_data)
    hevm.setEpoch(0, loop_count)
    timer_start = time.perf_counter_ns()
    hevm.run()
    execution_time = (time.perf_counter_ns() - timer_start) / pow(10, 9)
    # pow(10, 9) is to convert nanoseconds to seconds
    res = hevm.getOutput()
    rms = np.sqrt(np.mean(np.power(res[0] - ref_W, 2) + np.power(res[1] - ref_c, 2)))
    hevm.printer(execution_time, rms, loop_count)
