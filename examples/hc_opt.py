#!/usr/bin/env python

"""
@file hc_opt.py
@brief Hecate optimizer wrapper script for MLIR compilation.
"""

import hecate as hc
import os
import sys

if __name__ == "__main__":
    # Parse command line arguments
    argv = hc.hc_parser(__file__)
    compile_opt, waterline, benchmark, library, device, epochs, input_data = argv

    num_elements = len(input_data)
    epochs = int(epochs)
    HECATE = os.getenv("HECATE")
    print(HECATE)
    
    # Construct and execute the Hecate optimizer command
    # command = f"$HECATE/build/bin/hecate-opt --{compile_opt} \
    #         --ckks-config={HECATE}/profiled_{library}_{device}.json \
    #         --waterline={waterline} \
    #         --enable-debug-printer \
    #         {HECATE}/examples/traced/{benchmark}.mlir \
    #         --mlir-print-debuginfo \
    #         --mlir-pretty-debuginfo \
    #         --mlir-print-local-scope \
    #         --mlir-disable-threading \
    #         --mlir-print-ir-after-failure \
    #         -o {HECATE}/examples/optimized/{compile_opt}/{benchmark}.{waterline}.mlir \
    #         --mlir-timing"
    
    # if there is no --enable-debug-printer, ckks and earth dialect is not created
    command = f"$HECATE/build/bin/hecate-opt --{compile_opt} --ckks-config={HECATE}/profiled_{library}_{device}.json --waterline={waterline}  --enable-debug-printer {HECATE}/examples/traced/{benchmark}.mlir --mlir-print-debuginfo --mlir-pretty-debuginfo --mlir-print-local-scope --mlir-disable-threading --mlir-timing --mlir-print-ir-after-failure -o {HECATE}/examples/optimized/{compile_opt}/{benchmark}.{waterline}.mlir"
    os.system(command)