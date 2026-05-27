#!/usr/bin/env python

import torch
import hecate as hc

Empty = hc.Empty

# ------------------------------------------------------------------
# Utility Functions for Torch
# ------------------------------------------------------------------
# W * X + b :: (OUT_DIM, INTER_DIM) * (INTER_DIM, IN_DIM) + (OUT_DIM, IN_DIM) = (OUT_DIM, IN_DIM)
def linear_generate_random_data(
    OUT_DIM: int, INTER_DIM: int, IN_DIM: int, VALUE_RANGE: float
):
    weight = torch.empty(OUT_DIM, INTER_DIM, dtype=torch.double).uniform_(
        -VALUE_RANGE, VALUE_RANGE
    )
    bias = torch.zeros(OUT_DIM, IN_DIM, dtype=torch.double)
    input = torch.empty(INTER_DIM, IN_DIM, dtype=torch.double).uniform_(
        -VALUE_RANGE, VALUE_RANGE
    )
    return weight, bias, input


# X * W + b :: (IN_DIM, INTER_DIM) * (INTER_DIM, OUT_DIM) + (IN_DIM, OUT_DIM) = (IN_DIM, OUT_DIM)
def generate_random_data(IN_DIM: int, INTER_DIM: int, OUT_DIM: int, VALUE_RANGE: float):
    input = torch.empty(IN_DIM, INTER_DIM, dtype=torch.double).uniform_(
        -VALUE_RANGE, VALUE_RANGE
    )
    weight = torch.empty(INTER_DIM, OUT_DIM, dtype=torch.double).uniform_(
        -VALUE_RANGE, VALUE_RANGE
    )
    bias = torch.zeros(IN_DIM, OUT_DIM, dtype=torch.double)
    return input, weight, bias


# ------------------------------------------------------------------
# Data to JSON
# ------------------------------------------------------------------
def data_to_binary(input_tensor, weight_tensor, bias_tensor, file_name: str):
    data = {"input": input_tensor, "weight": weight_tensor, "bias": bias_tensor}
    file_path = f"{hc.hecate_dir}/examples/benchinputs/{file_name}.pt"
    torch.save(data, file_path)
    print(f"Data saved to {file_path}")


def binary_to_data(file_name: str):
    file_path = f"{hc.hecate_dir}/examples/benchinputs/{file_name}.pt"
    data = torch.load(file_path)
    print(f"Data loaded from {file_path}")

    return data["input"], data["weight"], data["bias"]


def data_to_json(input, weight, bias, file_name: str):
    data = {"input": input.tolist(), "weight": weight.tolist(), "bias": bias.tolist()}
    with open(f"{hc.hecate_dir}/examples/benchinputs/{file_name}.json", "w") as f:
        json.dump(data, f)
        print(f"Data saved to {hc.hecate_dir}/examples/benchinputs/{file_name}.json")


def json_to_data(file_name: str):
    with open(f"{hc.hecate_dir}/examples/benchinputs/{file_name}.json", "r") as f:
        data = json.load(f)
        print(f"Data loaded from {hc.hecate_dir}/examples/benchinputs/{file_name}.json")

    input = torch.tensor(data["input"], dtype=torch.double)
    weight = torch.tensor(data["weight"], dtype=torch.double)
    bias = torch.tensor(data["bias"], dtype=torch.double)

    return input, weight, bias


# ------------------------------------------------------------------
# Basic LLM Functions
# ------------------------------------------------------------------
def reshape_matrix(matrix: torch.Tensor, split: int):
    row, col = matrix.shape
    assert col % split == 0, f"Column size {col} must be divisible by split {split}"
    row, col = row * split, col // split
    return matrix.reshape(row, col)


def restore_matrix(matrix: torch.Tensor, split: int):
    row, col = matrix.shape
    assert row % split == 0, f"Row size {row} must be divisible by split {split}"
    row, col = row // split, col * split
    return matrix.reshape(row, col)
