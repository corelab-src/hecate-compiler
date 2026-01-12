#!/usr/bin/env python

import faulthandler

faulthandler.enable()

import torch
import numpy as np
from pathlib import Path

import hecate as hc
import poly.MPCB as MPCB
import poly.Func as Func
import hetorch as ht

# ------------------------------------------------------------------
# GEMV Function (Hecate Trace)
# ------------------------------------------------------------------
# a_slot_length = 2 ** (17-1) # HEaaN GPU, 2^16 = 65536
a_slot_length = 2 ** (16 - 1)  # HEONGPU, 2^15 = 32768
a_length, a_padded_length, a_epochs = None, None, None
a_weight, a_bias, a_input = None, None, None
source_path = Path(__file__).resolve()
source_dir = source_path.parent

fc_gemv_cp = ht.bsgsCP_orion
# (OUT_DIM, INTER_DIM) * (INTER_DIM, IN_DIM) + (OUT_DIM, IN_DIM) = (OUT_DIM, IN_DIM) by W^T * X^T + b^T
IN_DIM, INTER_DIM, OUT_DIM, VALUE_RANGE = 1, 64, 64, 1


# set ciphertext context
params_str = ",".join(["c"] * IN_DIM)
# print(f"params_str: {params_str}")


@hc.func(params_str)
def GEMV_orion(*ctxts):
    weight, bias = a_weight, a_bias
    slot_length = a_slot_length

    def act(input):
        return Func.HE_SiLU(input)

    output_array = []
    mpp_vec = np.empty((1,), dtype=object)
    for ctxt in ctxts:
        mpp_vec[0] = ctxt
        output = fc_gemv_cp(mpp_vec, weight, slot_length)
        output_array.append(output[0])

    return tuple(output_array)


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    argv = hc.hc_parser(__file__)
    compile_opt, waterline, benchmark, library, hardware, epochs, input_data = argv

    # generate input data
    input, weight, bias = ht.generate_random_data(
        IN_DIM, INTER_DIM, OUT_DIM, VALUE_RANGE
    )
    a_input, a_weight, a_bias = input, weight.T, bias.T
    ht.data_to_binary(a_input, a_weight, a_bias, benchmark)

    module_name = hc.save("traced", "traced")
    print(module_name, "traced module saved")
