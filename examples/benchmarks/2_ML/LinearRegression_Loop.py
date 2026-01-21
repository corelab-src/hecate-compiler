#!/usr/bin/env python
"""
@file Format.py
@brief Format benchmark using Hecate.

This file demonstrates the usage of Hecate to implement a format benchmark.
It includes command-line argument parsing via argparse and several helper functions.
"""

import faulthandler

faulthandler.enable()

import hecate as hc
import hetorch.stat as stat

# a_slot_length = 2 ** (17-1) # HEaaN GPU, 2^16 = 65536
a_slot_length = 2 ** (16 - 1)  # HEONGPU, 2^15 = 32768
a_length, a_padded_length, a_epochs = None, None, None


@hc.func("c,c,i")
def LinearRegression_Loop(x_data, y_data, epochs):
    """
    @brief Perform linear regression using gradient descent.

    This Hecate-traced function implements a simple linear regression using gradient descent.
    It takes input data and ground truth labels, computes the error, and updates the weight (W) and bias (b)
    over a specified number of epochs (iterations).

    @param x_data The input data.
    @param y_data The ground truth labels.
    @return A tuple (W, b) where W is the learned weight and b is the learned bias.
    """
    length = 4096
    W = hc.Plain([1.0])
    b = hc.Plain([0.0])

    learning_rate = hc.Plain([-0.01])

    with hc.loop(0, epochs, 1, inputarr=[W, b], num_elements=length) as i:
        print("length:", a_length)
        xW = x_data * W
        xWb = xW + b
        error = xWb - y_data
        errX = error * x_data
        meanErrX = errX * hc.Plain([1 / 2048])
        gradW = stat.sum_LRelements(meanErrX)
        meanErr = error * hc.Plain([1 / 2048])
        gradb = stat.sum_LRelements(meanErr)
        Wup = learning_rate * gradW
        bup = learning_rate * gradb
        W = W + Wup
        b = b + bup

    return W, b


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    """
    @brief Main entry point for tracing and saving the Hecate module.

    This function parses command-line arguments, then performs the Hecate module saving
    (which traces the defined functions), and finally prints the saved module name.
    """
    # aft_command : new-trace Format --opt dacapo --waterline 40 --library HEAAN --hardware GPU --epochs 2 --input True --padding True
    # aft_command : new-opt Format --opt dacapo --waterline 40 --library HEAAN --hardware GPU --epochs 2 --input True --padding True
    # aft_command : new-test Format --opt dacapo --waterline 40 --library HEAAN --hardware GPU --epochs 2 --input True --padding True

    # Parse command-line arguments
    argv = hc.hc_parser(__file__)
    compile_opt, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv
    epochs = int(loop_count)

    # Save the traced Hecate module and print the module name.
    modName = hc.save("traced", "traced")
    print(modName, "traced module saved")
