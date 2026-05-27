import hecate as hc 

import sys 
import time 
import numpy as np 

from random import * 
from pathlib import Path 

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv

a_compile_type = compile_type
a_compile_opt = int(waterline)
hc.setLibnHW(argv)

stem = Path(__file__).stem
hevm = hc.HEVM()
stem = Path(__file__).stem
print(stem)
hevm.load(f"traced/cst/_hecate_{stem}.cst", f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm")


seed(100)

unsorted_x = [uniform(-10, 10) for _ in range(8)]
print("Unsorted array: ", unsorted_x)
sorted_x = sorted(unsorted_x)
print("Sorted array: ", sorted_x)

hevm.setInput(0, unsorted_x)
timer = time.perf_counter_ns()
hevm.run()
timer = time.perf_counter_ns() - timer 
res = hevm.getOutput()

print("Encrypted Results: ")
power_sum = 0
for i in range(8):
    power_sum += np.power(res[i][0] - sorted_x[i], 2)
    print(res[i][0])
rms = np.sqrt(power_sum / 8)

# 2. Max Absolute Error 
max_abs_error = 0  

for i in range(8):
    current_error = np.abs(res[i][0] - sorted_x[i])

    if current_error > max_abs_error:
        max_abs_error = current_error

print("Max Absolute Error:", max_abs_error)

hevm.printer(timer/pow(10,9), rms)
