import hecate as hc 
import poly.Poly as poly 
from poly.Func import *



argv = hc.hc_parser(__file__)

compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv

# [0, 0, 1, 0, 0, ...]
def create_single_mask(i):
    mask_list = [0.0] * 8 # length is fixed to 8 
    if 0 <= i < 8:
        mask_list[i] = 1.0
    return hc.Plain(mask_list)

# arr: input ciphertxt, i: index to extract 
# output: ciphertxt with broadcasted i_val 
def extract_value(arr, i):
    rot = arr.rotate(i) 
    masked = rot * create_single_mask(0) 

    result = masked
    for j in [1, 2, 4]:  # log2(8) = 3 steps
        result = result + result.rotate(65536-j)
    return result

# get the min(a, b), max(a, b)
# a, b: -2048 ~2048 
def get_min_max(a, b):
    diff = a - b
    sum_val = a + b 
    normalization_factor = hc.Plain([1.0 / 100.0]) 
    normalized_diff = diff * normalization_factor
    sign = HE_sign(normalized_diff) * hc.Plain([2.0]) 
    min_val = (sum_val - diff * sign) * hc.Plain([0.5])
    max_val = (sum_val + diff * sign) * hc.Plain([0.5])
    return min_val, max_val

@hc.func("c")
def BubbleSort(arr):
    result = []
    for i in range(8): # arr length is fixed to 8 
        for j in range(0, 8-1-i):
            a_val = extract_value(arr, j)
            b_val = extract_value(arr, j+1)
            min_val, max_val = get_min_max(a_val, b_val)
            # min_val should be placed in index j 
            # max_val should be placed in index j+1
            mask_min = create_single_mask(j)
            mask_max = create_single_mask(j+1)
            mask_all = [1.0] * 8 
            mask_all[j] = 0.0
            mask_all[j+1] = 0.0
            mask = hc.Plain(mask_all)
            arr_result = arr * mask + min_val * mask_min + max_val * mask_max
            arr = arr_result

    for i in range(8):
        i_val = extract_value(arr, i) # this will return a ciphertext of the broadcasted index value 
        result.append(i_val)

    return result # this will return an array that is index-available 

modName = hc.save("traced", "traced")
print (modName)
