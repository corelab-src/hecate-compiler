#!/usr/bin/env python
"""
@file parser.py
@brief Command-line argument parser for Hecate benchmarks.
"""

# ------------------------------------------------------------------
# Memory Measurement
# ------------------------------------------------------------------
import psutil

def print_mem(label=""): # CPU memory measurement
    proc = psutil.Process(os.getpid())
    mem_gb = proc.memory_info().rss / (1024 ** 3)
    print(f"{label} ▶ {mem_gb:.2f} GB")
    return mem_gb

import numpy as np

# ------------------------------------------------------------------
# Command-Line Argument Parsing
# ------------------------------------------------------------------
import argparse
import json
import os
from random import uniform, seed

# set and initialize global variables
input_length, padded_length, epochs = 0, 0, 0

def hc_parser(command_path):
    global input_length, padded_length, epochs
    """
    @brief Parse command-line arguments.

    This function creates an ArgumentParser object to process command-line inputs for hc-trace and hc-test.
    It handles the following arguments:
    benchmark: Specifies the benchmark to run.
    --opt: Specifies the compile type.
    --waterline: Specifies the waterline.
    --library: Specifies the library.
    --hardware: Specifies the hardware.
    --epochs: Specifies the number of epochs.
    --input: Specifies the input data.
    --split: Specifies whether to split the input data into parts.
    --padding: Specifies whether to pad the input data.
    
    @return A list containing the parsed and pre-processed arguments (opt, waterline, benchmark, library, hardware, epochs, padded_data).
    """
    # Create an ArgumentParser instance with a description.
    # archived_command : hc-test <$1 sys.argv[1] compile_type> <$2 sys.argv[2] waterline:int> <$3 sys.argv[3] benchmark:str> <$4 sys.argv[3] library:"HEAAN, SEAL, HEONGPU"> <$5 sys.argv[4] hardware:"GPU, CPU"> <$6 sys.argv[5] epochs:int>
    # new_command : hc-trace <$1 sys.argv[1] benchmark:str> <$2 sys.argv[2] compile_type:str> <$3 sys.argv[3] waterline:int> <$4 sys.argv[4] library:str> <$5 sys.argv[5] hardware:str> <$6 sys.argv[6] epochs:int> <$7 sys.argv[7] input:str> <$8 sys.argv[8] padding:str> <$9 sys.argv[9] split:int>
    
    parser = argparse.ArgumentParser(description="hc-trace and hc-test argument parser")
    # Positional argument: benchmark
    parser.add_argument("benchmark", type=str, nargs="?", default="Format", help="Benchmark name, default is Format")
    # Optional arguments
    # parser.add_argument("--opt", type=str, default="dacapo", help="Compile type")
    parser.add_argument("--opt", type=str, required=True, help="Compile type") # temporary required
    parser.add_argument("--waterline", type=int, default=40, help="Waterline")
    parser.add_argument("--library", type=str, default="HEONGPU", help="Library")
    parser.add_argument("--device", type=str, default="GPU", choices=["GPU", "CPU"], help="Device")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--input", type=str, default="not used",
                        help="If set to 'True', load input data from the specific folder's benchmark_name.json file; otherwise")
    # split: dimension split
    parser.add_argument("--split", type=int, default=0,
                        help="If set to 'not 0', split the input data into n parts; otherwise, default to 1")
    parser.add_argument("--padding", type=str, default="False",
                        help="If set to 'True', pad the input data to the next power of 2; otherwise, default to False")
    # Todo: multi input data
    args, unknown = parser.parse_known_args()

    # set input data
    seed(100)
    
    if not isinstance(args.library, str):
        parser.error("Error: Library is not a string, Choices are: HEAAN, SEAL, HEONGPU")
    else:
        args.library = args.library.upper()
    
    if args.input.lower() == "not used":
        print("This benchmark does not use input data, inherit from parent class")
    elif args.input.lower() == "false":
        print(f"Input option = False. Use random input data in {args.benchmark} file")
        print(f"If there is no input data in {args.benchmark} file, use (4096, -1, 1) random data")
        input_data = [uniform(-1, 1) for _ in range(4096)]
        input_data = np.array(input_data)
    elif args.input.lower() == "true":
        # load input data from file
        file_path = os.path.join(os.getcwd(), "benchinputs", f"{args.benchmark}.json")   
        if os.path.exists(file_path) and os.path.isfile(file_path):
            with open(file_path, 'r') as file_content:
                input_data = json.load(file_content)
                if args.split != 0:
                    # split input data into n parts
                    input_data = np.array(input_data)
                    input_data = np.split(input_data, args.split, axis=1)
                else:
                    input_data = np.array(input_data)
            print(f"Loaded input data from file: {file_path} \n")
        else:
            parser.error(f"Error: Input data file {file_path} does not exist")
    elif args.input.isdigit():
        # use input data as a number
        print(f"Input option is a number. Use {args.input} as input data")
        input_data = [uniform(-1, 1) for _ in range(int(args.input))]
        print(f"input_data= [uniform(-1, 1) for _ in range({int(args.input)})] \n")
        input_data = np.array(input_data)
    elif args.input.startswith("[") and args.input.endswith("]"):
        # use input data as a JSON array
        try:
            input_data = json.loads(args.input)
        except json.JSONDecodeError:
            parser.error("Error: Input data is not a valid JSON array")
        input_data = np.array(input_data)
        print(f"Input data is directly set \n")
        print("input_data:", input_data)
    elif args.input.startswith("(") and args.input.endswith(")"):
        # use input data as a tuple, (num, min, max)
        inner = args.input[1:-1]
        num, min, max = inner.split(",")
        input_data = generate_random_data_from_tuple(int(num), float(min), float(max))
        print(f"Input data is directly set \n")
        print("input_data:", input_data)
    else:
        parser.error("Error: Input command is not valid")
    
    # flatten and pad input data, padding is optional
    global input_length, epochs, padded_length
    padded_data = []
    if args.input.lower() == "not used":
        input_length = 0
        padded_length = 0
        raw_shape = ()
    else:
        if args.split != 0:
            for i in range(args.split):
                flat_data, raw_shape = flatten_data(input_data[i])
                if args.padding.lower() == "true":
                    padded_data.append(pad_to_multiple(flat_data))
                else:
                    padded_data.append(flat_data)
            padded_length = len(padded_data[0])
        else:
            flat_data, raw_shape = flatten_data(input_data)
            if args.padding.lower() == "true":
                padded_data = pad_to_multiple(flat_data)
            else:
                padded_data = flat_data
            padded_length = len(padded_data)
        
        input_length = len(flat_data)
    epochs = args.epochs

    # convert arguments to before_command format
    argv = [args.opt, args.waterline, args.benchmark, args.library, args.device, args.epochs, padded_data]    
    # print command and arguments
    print(f"{'Benchmark':<25}  {'Opt':<25} {'Waterline':<12}")
    print(f"{args.benchmark:<25}  {args.opt:<25} {args.waterline:<12}\n")
    print(f"{'Library':<12}  {'Device':<12} {'Epochs':<12} {'Input':<12}")
    print(f"{args.library:<12}  {args.device:<12} {args.epochs:<12} {args.input:<12}\n")
    # print(f"{'Split':<12}  {'Padding':<12}")
    # print(f"{args.split:<12}  {args.padding:<12}\n")
    # print(f"{'args.split':<12} {'Padded shape':<12}  {'Flattened shape':<16}  {'Raw shape'}")
    # if args.split != 0:
    #     for i in range(args.split):
    #         print(f"{i:<12} {len(padded_data[i]):<12}  {len(flat_data):<16}  {raw_shape}")
    # else:
    #     print(f"{'0':<12} {padded_length:<12}  {input_length:<16}  {raw_shape}")
    # print("")
    # print("input_length:", input_length)
    print("================================================")
    print("arguments parsing and data pre-processing done")
    print("================================================")

    return argv

# ------------------------------------------------------------------
# Data Manipulation
# ------------------------------------------------------------------
def generate_random_data_from_tuple(num, min, max):
    """
    @brief Generate random data from a tuple.
    """
    return [uniform(min, max) for _ in range(num)]

def flatten_data(data):
    """
    @brief Flatten an n-dimensional list into a 1D vector while preserving its original shape.
    
    This function converts an n-dimensional list into a 1D list (vector) and returns the original shape.
    It ensures that the input data is a numpy array of type float64.
    The conversion using tolist() is necessary because sometimes the input data might be a torch.Tensor,
    and tolist() converts it into a native Python list for consistent behavior.
    
    @param data: An n-dimensional list representing the input data.
    @return: A tuple (flat_vector, shape) where flat_vector is the flattened 1D list and shape is the original shape tuple.
    """
    # code from python / hecate / hecate / expr.py / class Plain(Expr)
    # If data is not a numpy array, convert it to one with dtype float64.
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float64)
    # Even if data is a numpy array, convert it to a list and then back to a numpy array
    # to ensure that any tensor-like objects (e.g., torch.Tensor) are properly handled.
    if isinstance(data, np.ndarray):
        shape = data.shape
        data = np.array(data.tolist(), dtype=np.float64)
    
    flat_vector = data.flatten().tolist()
    return flat_vector, shape

def pad_to_multiple(data):
    """
    @brief Pads the input data with zeros so its length is a power of '2'.
    
    When encrypting, Hecate copies the input data to fill the ciphertext vector.
    If the length is not a multiple of the desired value, the reduction via rotations may not work as expected.
    This function pads the data with zeros accordingly.
    
    @param data: List of float values representing the input data.
    @return: A new list where the length is padded to be a power of 2.
    """
    # Find next power of 2
    length = len(data)
    next_power = 1
    while next_power < length:
        next_power *= 2
        
    # Pad with zeros if needed
    if length < next_power:
        pad_length = next_power - length
        data = data + [0.0] * pad_length
    return data

def restore_data(flat_vector, shape):
    """
    @brief Restore a flattened 1D vector back into its original n-dimensional shape.
    
    Given a flat vector and the original shape, this function reconstructs the n-dimensional list.
    If extra padded zeros are present, they are discarded.
    
    @param flat_vector: The 1D list representing the flattened data (possibly with extra padded zeros).
    @param shape: A tuple representing the original shape.
    @return: An n-dimensional list restored to its original configuration.
    """
    original_size = int(np.prod(shape))
    truncated_vector = flat_vector[:original_size]
    restored = np.array(truncated_vector).reshape(shape).tolist()
    print(f"Restored data shape: {len(restored)}")
    return restored

def copy_first_slot(data):
    """
    @brief Copy the first slot value to all slots in the ciphertext.
    
    @param data: The encrypted data.
    @return: Data with the first slot value copied to all slots.
    """
    mask = [1.0] + [0.0] * (padded_length - 1)
    result = data * mask
    for i in range(log_2(padded_length)):
        rot = result.rotate(1 << i)
        result = result + rot
    return result

def make_mask(length):
    """
    @brief Create a mask with 1s for the first 'length' slots and 0s for the rest.
    
    @param length: Number of slots to set to 1.
    @return: A mask array.
    """
    mask = [1.0] * length + [0.0] * (padded_length - length)
    return mask

def masking_data(data, length):
    """
    @brief Apply a mask to the data to zero out slots beyond a certain length.
    
    @param data: The data to mask.
    @param length: Number of slots to keep.
    @return: Masked data.
    """
    mask = make_mask(length)
    masked_data = data * mask
    return masked_data

# ------------------------------------------------------------------
# Basic Statistic Functions
# ------------------------------------------------------------------
def log_2(num_elements):
    """
    @brief Calculate the base-2 logarithm of a number.
    
    @param num_elements: The number to calculate log_2 for.
    @return: The integer log_2 value.
    """
    power = 0
    while num_elements > 1:
        num_elements = num_elements // 2
        power += 1
    return power

# ------------------------------------------------------------------
# Time Measurement
# ------------------------------------------------------------------
import time

def measure(func, *args, **kwargs):
    """
    @brief Measure the execution time of a function.
    
    @param func: Function to measure.
    @param *args: Arguments to pass to the function.
    @param **kwargs: Keyword arguments to pass to the function.
    @return: A tuple containing the function's return value and the execution time in seconds.
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    print(f"[measure] {func.__name__}")
    print(f"args={args}, kwargs={kwargs}")
    print(f"executed in {elapsed:.6f}s")
    return result, elapsed