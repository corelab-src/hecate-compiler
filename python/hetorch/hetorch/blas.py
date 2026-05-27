#!/usr/bin/env python
import numpy as np
import math


def roll(input, i):
    return input.rotate(-i)  # rotate to lower index

def cint(i):
    return int(np.ceil(i))

def fint(i):
    return int(np.floor(i))

# Utility Functions for hetorch
def masking_ciphertext(ctxt, input_length, slot_length):
    mask_input = np.zeros(slot_length, dtype=np.double)
    mask_input[:input_length].fill(1)
    ctxt[0] = ctxt[0] * mask_input
    return ctxt

# torch.nn.functional.linear
# nn.Linear() = x * weight.T + bias
# # 2D tensor (1024,4096) :: A[0] = (4096, ):: 1D vector
# # ------------------------ (column) = 4096
# # |
# # |
# # |
# # | (row) = 1024

# Global Parameters
hetorch_bsgs_ratio = 2

# ------------------------------------------------------------------
# gemvCP Functions
# ------------------------------------------------------------------
# linear = weight * input + bias
def linear(input, weight, bias, nt):
    # input padding and scaling
    result = np.full((1), Empty(), dtype=object)  # shape (1,)
    outdim, indim = weight.shape
    S = torch.ones(indim)  # shape (indim,)
    S = F.pad(S, (0, nt - S.shape[0]), mode="constant", value=0)  # shape (nt,)
    input[0] = S * input[0]  # shape (nt,)
    input[0] = input[0] + roll(
        input[0], indim
    )  # shape (nt,), data (in_dim, 0 padding, in_dim)

    # weight mapping and padding
    it = (indim + outdim - 1) // outdim  # = np.ceil(indim / outdim)
    weight = torch.stack(
        [torch.roll(weight[i, :], -i) for i in range(outdim)]
    )  # shape (outdim, indim), data just rotated to right
    weight = F.pad(
        weight, (0, it * outdim - indim), mode="constant", value=0
    )  # shape (outdim, it*outdim)
    weight = einops.rearrange(
        weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=outdim, i2=it, i3=outdim
    )  # transpose
    weight = F.pad(
        weight, (0, nt - weight.shape[1]), mode="constant", value=0
    )  # shape (outdim, nt)

    # rotation and point-wise multiplication
    for i in range(outdim):
        result[0] = result[0] + roll(input[0], -i) * weight[i, :]  # right rotation

    # self-sum
    for j in range(cint(np.log2(it))):
        result[0] = result[0] + roll(result[0], -pow(2, j) * outdim)

    # bias padding and addition
    bias = F.pad(bias, (0, nt - bias.shape[0]), mode="constant", value=0)
    result[0] = result[0] + bias
    return result

def gemvCP_linear(input, weight, slot_length):
    # input padding and scaling
    result = np.full((1), Empty(), dtype=object)  # shape (1,)
    outdim, indim = weight.shape
    S = torch.ones(indim)  # shape (indim,)
    S = F.pad(S, (0, slot_length - S.shape[0]), mode="constant", value=0)  # shape (slot_length,)
    input[0] = S * input[0]  # shape (slot_length,)
    input[0] = input[0] + roll(
        input[0], indim
    )  # shape (slot_length,), data (in_dim, 0 padding, in_dim)

    # weight mapping and padding
    it = (indim + outdim - 1) // outdim  # = np.ceil(indim / outdim)
    weight = torch.stack(
        [torch.roll(weight[i, :], -i) for i in range(outdim)]
    )  # shape (outdim, indim), data just rotated to right
    weight = F.pad(
        weight, (0, it * outdim - indim), mode="constant", value=0
    )  # shape (outdim, it*outdim)
    weight = einops.rearrange(
        weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=outdim, i2=it, i3=outdim
    )  # transpose
    weight = F.pad(
        weight, (0, slot_length - weight.shape[1]), mode="constant", value=0
    )  # shape (outdim, slot_length)

    # rotation and point-wise multiplication
    for i in range(outdim):
        result[0] = result[0] + roll(input[0], -i) * weight[i, :]  # right rotation

    # self-sum
    for j in range(cint(np.log2(it))):
        result[0] = result[0] + roll(result[0], -pow(2, j) * outdim)
    return result