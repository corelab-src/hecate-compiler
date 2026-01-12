#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

import numpy as np
from pathlib import Path
import os
import math

import hecate as hc

Empty = hc.Empty

from hetorch.blas import (
    roll,
    cint,
    fint,
    masking_ciphertext,
    full_replicate_ciphertext,
    replicate_ciphertext,
)

# Constants
slot_length = 2**15
N = 2**15  # Number of slots
orion_bsgs_ratio = 2


# ------------------------------------------------------------------
# Orion BSGS Functions
# ------------------------------------------------------------------
def BSGSIndex(nonZeroDiags, slots, N1):
    index = {}
    rotN1Map = {}
    rotN2Map = {}

    for rot in nonZeroDiags:
        idxN1 = ((rot // N1) * N1) % slots
        idxN2 = rot % N1

        if idxN1 not in index:
            index[idxN1] = []
        if idxN2 not in index[idxN1]:
            index[idxN1].append(idxN2)

        rotN1Map[idxN1] = True
        rotN2Map[idxN2] = True

    for k in index:
        index[k].sort()

    return index, sorted(rotN1Map.keys()), sorted(rotN2Map.keys())


def find_best_bsgs_ratio(nonZeroDiags, maxN, bsgs_ratio):
    min_cost = float("inf")
    best_n1 = 1

    for N1 in [2**i for i in range(int(math.log2(maxN)) + 1)]:
        if N1 == 0:
            continue

        _, rotN1, rotN2 = BSGSIndex(nonZeroDiags, maxN, N1)
        if not rotN1:
            continue

        # prevent zero division
        nbN1 = len(rotN1)
        nbN2 = len(rotN2)
        cost = (nbN1 - 1) + (nbN2 - 1)

        if cost < min_cost:
            min_cost = cost
            best_n1 = N1

        if bsgs_ratio != 0:
            if (nbN2 / nbN1) > bsgs_ratio:
                return N1 // 2 if N1 > 1 else 1
                # return best_n1
            elif (nbN2 / nbN1) == bsgs_ratio:
                return N1

    return best_n1


def orion_cal_bsgs_list(out_dim, shift_list, bsgs_ratio):
    if not shift_list:
        return 0, 0, [], {}, []
    # print(f"out_dim: {out_dim}, bsgs_ratio: {bsgs_ratio}, len(shift_list): {len(shift_list)}")
    # print(f"shift_list: {shift_list}")
    # Find the optimal N1 using the Lattigo algorithm
    optimal_n1 = find_best_bsgs_ratio(shift_list, out_dim, bsgs_ratio)

    # Generate the final mapping with the optimal N1
    index, giant_steps, baby_steps = BSGSIndex(shift_list, out_dim, optimal_n1)

    optimized_mapping = {}
    groups = []

    # Ensure all shifts are mapped
    for shift in shift_list:
        gs = ((shift // optimal_n1) * optimal_n1) % out_dim
        bs = shift % optimal_n1
        optimized_mapping[shift] = (gs, bs)

    # Organize into groups
    for gs in sorted(set(g for g, _ in optimized_mapping.values())):
        group = []
        for shift in shift_list:
            if optimized_mapping[shift][0] == gs:
                group.append(shift)
        if group:
            groups.append(sorted(group))

    best_baby = len(baby_steps)
    best_giant = len(giant_steps)

    return best_baby, best_giant, groups, optimized_mapping, baby_steps


def masked_bsgsCP_orion(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    # print(f"masked_bsgsCP_orion: weight (out_dim, inter_dim): {weight.shape}")
    mask_input = torch.ones(inter_dim)  # shape (inter_dim,)
    mask_input = F.pad(
        mask_input, (0, slot_length - mask_input.shape[0]), mode="constant", value=0
    )  # shape (slot_length,)
    input[0] = mask_input * input[0]  # shape (slot_length,)
    return bsgsCP_orion(input, weight, slot_length, bsgs_ratio)


def bsgsCP_orion(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    # print(f"bsgsCP_orion: weight (out_dim, inter_dim): {weight.shape}")
    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))

    # new_inter_dim = inter_dim if is_inter_dim_power_of_two else 2 ** cint(np.log2(inter_dim))
    new_inter_dim = 32768
    new_out_dim = out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
    weight = F.pad(
        weight,
        (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
        mode="constant",
        value=0,
    )  # (out_dim, expand*out_dim)
    out_dim, inter_dim = weight.shape
    # print(f"After padding weight.shape: {weight.shape}")

    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint((out_dim // 2) / inter_dim) + 1
            input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
            weight1 = weight[: out_dim // 2, :]
            weight2 = weight[out_dim // 2 :, :]
            result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
            result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
            masked_result1 = masking_ciphertext(result1, out_dim // 2, slot_length)
            masked_result2 = masking_ciphertext(
                result2, out_dim - out_dim // 2, slot_length
            )
            return masked_result1 + roll(masked_result2[0], out_dim // 2)
        expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        # print(f"inter_dim {inter_dim} == slot_length, out_dim: {out_dim}")
        if inter_dim % out_dim != 0:
            expand = 1
            weight = weight.T
            weight = F.pad(
                weight, (0, inter_dim - out_dim), mode="constant", value=0
            )  # (out_dim, expand*out_dim)
        else:
            weight = einops.rearrange(
                weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
            )  # transpose
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
        # print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if is_out_dim_bigger:
        dimension_for_shift = inter_dim
    elif inter_dim == slot_length:
        dimension_for_shift = inter_dim if inter_dim % out_dim != 0 else out_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = roll(input[0], -bs) * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_fc(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    print(f"bsgsCP_orion_fc: weight (out_dim, inter_dim): {weight.shape}")
    # mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    # mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    # input[0] = mask_input * input[0] # shape (slot_length,)

    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))
    new_inter_dim = 32768
    new_out_dim = out_dim if is_out_dim_power_of_two else 32768
    weight = F.pad(
        weight,
        (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
        mode="constant",
        value=0,
    )  # (out_dim, expand*out_dim)
    out_dim, inter_dim = weight.shape

    print(f"bsgsCP_orion_fc: weight (out_dim, inter_dim): {weight.shape}")
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint(out_dim / inter_dim)  # full replicate
        else:
            expand_inter = cint(out_dim / inter_dim) + 1
        # elif out_dim > slot_length/2 :
        # expand_inter = cint((out_dim//2) / inter_dim) + 1
        # input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # weight1 = weight[:out_dim//2, :]
        # weight2 = weight[out_dim//2:, :]
        # result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
        # result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
        # masked_result1 = masking_ciphertext(result1, out_dim//2, slot_length)
        # masked_result2 = masking_ciphertext(result2, out_dim - out_dim//2, slot_length)
        # return masked_result1 + roll(masked_result2[0], out_dim//2)
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        print(f"inter_dim {inter_dim} == slot_length, out_dim: {out_dim}")
        if inter_dim % out_dim != 0:
            expand = 1
            weight = weight.T
            weight = F.pad(
                weight, (0, inter_dim - out_dim), mode="constant", value=0
            )  # (out_dim, expand*out_dim)
        else:
            weight = einops.rearrange(
                weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
            )  # transpose
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
    print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if out_dim == inter_dim == slot_length:
        dimension_for_shift = out_dim
    elif is_out_dim_bigger or (inter_dim == slot_length and inter_dim % out_dim != 0):
        dimension_for_shift = inter_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_ptwo(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    print(f"bsgsCP_orion_ptwo: weight (out_dim, inter_dim): {weight.shape}")
    # mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    # mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    # input[0] = mask_input * input[0] # shape (slot_length,)

    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))
    new_inter_dim = 32768
    new_out_dim = out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
    weight = F.pad(
        weight,
        (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
        mode="constant",
        value=0,
    )  # (out_dim, expand*out_dim)
    out_dim, inter_dim = weight.shape

    print(f"bsgsCP_orion_ptwo: weight (out_dim, inter_dim): {weight.shape}")
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint(out_dim / inter_dim)  # full replicate
        #     expand_inter = cint((out_dim//2) / inter_dim) + 1
        #     input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        #     weight1 = weight[:out_dim//2, :]
        #     weight2 = weight[out_dim//2:, :]
        #     result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
        #     result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
        #     masked_result1 = masking_ciphertext(result1, out_dim//2, slot_length)
        #     masked_result2 = masking_ciphertext(result2, out_dim - out_dim//2, slot_length)
        #     return masked_result1 + roll(masked_result2[0], out_dim//2)
        else:
            expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        print(f"inter_dim {inter_dim} == slot_length, out_dim: {out_dim}")
        if inter_dim % out_dim != 0:
            expand = 1
            weight = weight.T
            weight = F.pad(
                weight, (0, inter_dim - out_dim), mode="constant", value=0
            )  # (out_dim, expand*out_dim)
        else:
            weight = einops.rearrange(
                weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
            )  # transpose
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
    print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if out_dim == inter_dim == slot_length:
        dimension_for_shift = out_dim
    elif is_out_dim_bigger or (inter_dim == slot_length and inter_dim % out_dim != 0):
        dimension_for_shift = inter_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_mlp(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))
    new_inter_dim = (
        inter_dim if is_inter_dim_power_of_two else 2 ** cint(np.log2(inter_dim))
    )
    if inter_dim == 784:
        new_inter_dim = 8192
    new_out_dim = out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
    weight = F.pad(
        weight,
        (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
        mode="constant",
        value=0,
    )  # (out_dim, expand*out_dim)
    out_dim, inter_dim = weight.shape

    print(f"bsgsCP_orion_mlp: weight (out_dim, inter_dim): {weight.shape}")
    # mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    # mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    # input[0] = mask_input * input[0] # shape (slot_length,)
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint((out_dim // 2) / inter_dim) + 1
            input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
            weight1 = weight[: out_dim // 2, :]
            weight2 = weight[out_dim // 2 :, :]
            result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
            result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
            masked_result1 = masking_ciphertext(result1, out_dim // 2, slot_length)
            masked_result2 = masking_ciphertext(
                result2, out_dim - out_dim // 2, slot_length
            )
            return masked_result1 + roll(masked_result2[0], out_dim // 2)
        expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        print(f"inter_dim == slot_length, out_dim: {out_dim}, inter_dim: {inter_dim}")
        expand = 1
        weight = weight.T
        weight = F.pad(
            weight, (0, inter_dim - out_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
        print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if out_dim == inter_dim == slot_length:
        dimension_for_shift = out_dim
    elif is_out_dim_bigger or inter_dim == slot_length:
        dimension_for_shift = inter_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_lola(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))
    if not (is_inter_dim_power_of_two and is_out_dim_power_of_two):
        # new_inter_dim = inter_dim if is_inter_dim_power_of_two else 2 ** cint(np.log2(inter_dim))
        new_inter_dim = 8192
        if out_dim == 10:
            new_out_dim = 8192
        else:
            new_out_dim = (
                out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
            )
        # new_out_dim = out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
        weight = F.pad(
            weight,
            (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
            mode="constant",
            value=0,
        )  # (out_dim, expand*out_dim)
        out_dim, inter_dim = weight.shape

    print(f"Orion: bsgsCP_orion_fc weight (out_dim, inter_dim): {weight.shape}")
    # mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    # mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    # input[0] = mask_input * input[0] # shape (slot_length,)
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint((out_dim // 2) / inter_dim) + 1
            input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
            weight1 = weight[: out_dim // 2, :]
            weight2 = weight[out_dim // 2 :, :]
            result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
            result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
            masked_result1 = masking_ciphertext(result1, out_dim // 2, slot_length)
            masked_result2 = masking_ciphertext(
                result2, out_dim - out_dim // 2, slot_length
            )
            return masked_result1 + roll(masked_result2[0], out_dim // 2)
        expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        print(f"inter_dim == slot_length, out_dim: {out_dim}, inter_dim: {inter_dim}")
        expand = 1
        weight = weight.T
        weight = F.pad(
            weight, (0, inter_dim - out_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
        print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if out_dim == inter_dim == slot_length:
        dimension_for_shift = out_dim
    elif is_out_dim_bigger or inter_dim == slot_length:
        dimension_for_shift = inter_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_lenet(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    is_inter_dim_power_of_two = inter_dim == 2 ** cint(np.log2(inter_dim))
    is_out_dim_power_of_two = out_dim == 2 ** cint(np.log2(out_dim))
    if not (is_inter_dim_power_of_two and is_out_dim_power_of_two):
        # new_inter_dim = inter_dim if is_inter_dim_power_of_two else 2 ** cint(np.log2(inter_dim))
        new_inter_dim = (
            inter_dim if is_inter_dim_power_of_two else 2 ** cint(np.log2(inter_dim))
        )
        new_out_dim = (
            out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
        )
        # new_out_dim = out_dim if is_out_dim_power_of_two else 2 ** cint(np.log2(out_dim))
        weight = F.pad(
            weight,
            (0, new_inter_dim - inter_dim, 0, new_out_dim - out_dim),
            mode="constant",
            value=0,
        )  # (out_dim, expand*out_dim)
        out_dim, inter_dim = weight.shape

    print(f"Orion: bsgsCP_orion_fc weight (out_dim, inter_dim): {weight.shape}")
    # mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    # mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    # input[0] = mask_input * input[0] # shape (slot_length,)
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger:
        if out_dim > slot_length / 2:
            expand_inter = cint((out_dim // 2) / inter_dim) + 1
            input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
            weight1 = weight[: out_dim // 2, :]
            weight2 = weight[out_dim // 2 :, :]
            result1 = bsgsCP_orion_core(input, weight1, slot_length, bsgs_ratio)
            result2 = bsgsCP_orion_core(input, weight2, slot_length, bsgs_ratio)
            masked_result1 = masking_ciphertext(result1, out_dim // 2, slot_length)
            masked_result2 = masking_ciphertext(
                result2, out_dim - out_dim // 2, slot_length
            )
            return masked_result1 + roll(masked_result2[0], out_dim // 2)
        expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)

    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = weight.T
    elif inter_dim == slot_length:
        print(f"inter_dim == slot_length, out_dim: {out_dim}, inter_dim: {inter_dim}")
        expand = 1
        weight = weight.T
        weight = F.pad(
            weight, (0, inter_dim - out_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
    else:
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
        print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    if out_dim == inter_dim == slot_length:
        dimension_for_shift = out_dim
    elif is_out_dim_bigger or inter_dim == slot_length:
        dimension_for_shift = inter_dim
    else:
        dimension_for_shift = out_dim
    for i in range(dimension_for_shift):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    cal_bsgs = out_dim if not (inter_dim == slot_length) else inter_dim
    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(cal_bsgs, shift_list, bsgs_ratio)
    )

    padded_weight = torch.zeros((dimension_for_shift, slot_length), dtype=torch.double)
    if inter_dim == slot_length:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, :] = torch.roll(weight[i, :], offset)
            else:
                padded_weight[i, :] = 0
    else:
        for i in range(dimension_for_shift):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    if expand > 1:
        for j in range(cint(np.log2(expand))):
            result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


def bsgsCP_orion_core(input, weight, slot_length, bsgs_ratio=orion_bsgs_ratio):
    out_dim, inter_dim = weight.shape
    print(f"bsgsCP_orion_core: weight (out_dim, inter_dim): {weight.shape}")
    is_out_dim_bigger = out_dim > inter_dim
    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger:
        weight = torch.stack(
            [torch.roll(weight[i, :], -(i % inter_dim)) for i in range(out_dim)]
        )  # diagonal, same dimension (out_dim, inter_dim)
        weight = weight.T  # (inter_dim, out_dim)
    else:
        if not (inter_dim > slot_length / 2):
            input[0] = input[0] + roll(input[0], inter_dim)  # shape (slot_length,)
        else:
            assert (
                inter_dim == slot_length
            ), f"inter_dim: {inter_dim}, slot_length: {slot_length}"
        weight = torch.stack(
            [torch.roll(weight[i, :], -i) for i in range(out_dim)]
        )  # diagonal, (out_dim, inter_dim)
        weight = F.pad(
            weight, (0, expand * out_dim - inter_dim), mode="constant", value=0
        )  # (out_dim, expand*out_dim)
        weight = einops.rearrange(
            weight, "i1 (i2 i3) -> i3 (i2 i1)", i1=out_dim, i2=expand, i3=out_dim
        )  # transpose
        print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")

    shift_list = []
    for i in range(out_dim if not is_out_dim_bigger else inter_dim):
        if torch.any(weight[i, :] != 0):
            shift_list.append(i)

    baby_step, giant_step, groups, best_shift_mapping, baby_step_list = (
        orion_cal_bsgs_list(out_dim, shift_list, bsgs_ratio)
    )

    if out_dim == slot_length:
        padded_weight = weight
    else:
        padded_weight = torch.zeros(
            (out_dim if not is_out_dim_bigger else inter_dim, slot_length),
            dtype=torch.double,
        )
        for i in range(out_dim if not is_out_dim_bigger else inter_dim):
            if i in shift_list:
                (offset, _) = best_shift_mapping[i]
                padded_weight[i, offset : offset + expand * out_dim] = weight[i, :]
            else:
                padded_weight[i, :] = 0

    # inplace operation
    group_offset = 0
    tmp = np.full((1,), Empty(), dtype=object)  # shape (1,)
    result = np.full((1,), Empty(), dtype=object)  # shape (1,)
    for idx, group in enumerate(groups):
        for jdx, shift in enumerate(group):
            (offset, bs) = best_shift_mapping[shift]
            if idx != 0 and jdx == 0:
                if not isinstance(result[0], Empty):
                    result[0] = roll(result[0], (offset - group_offset))
            group_offset = offset
            if torch.any(padded_weight[shift, :] != 0):
                # print(f"padded_weight[{shift}, :] != 0 , (offset, bs): ({offset}, {bs})")
                if bs == 0:
                    tmp[0] = input[0] * padded_weight[shift, :]
                else:
                    tmp[0] = tmp[0] + roll(input[0], -bs) * padded_weight[shift, :]
        if not isinstance(tmp[0], Empty):
            if isinstance(result[0], Empty):
                result[0] = tmp[0]
            else:
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty()  # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    # self-sum
    for j in range(cint(np.log2(expand))):
        result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result


# ------------------------------------------------------------------
# Orion Functions
# ------------------------------------------------------------------
conv_gemv_cp = masked_bsgsCP_orion


def create_toeplitz_matrix(kernel, input_h, input_w, stride, padding):
    out_c, in_c, ker_h, ker_w = kernel.shape

    # Calculate output dimensions
    out_h = (input_h + 2 * padding - ker_h) // stride + 1
    out_w = (input_w + 2 * padding - ker_w) // stride + 1
    output_shape = (out_c, out_h, out_w)

    # Initialize the Toeplitz matrix with zeros
    num_rows = out_c * out_h * out_w
    num_cols = in_c * input_h * input_w
    toeplitz_tensor = torch.zeros((num_rows, num_cols), dtype=kernel.dtype)
    print("kernel.shape -> toeplitz.shape: ", kernel.shape, "->", toeplitz_tensor.shape)

    # Populate the Toeplitz matrix
    for oc in range(out_c):
        for oh in range(out_h):
            for ow in range(out_w):
                # Calculate the current row index in the Toeplitz matrix
                row_idx = oc * (out_h * out_w) + oh * out_w + ow

                # For each input channel and kernel element
                for ic in range(in_c):
                    for kh in range(ker_h):
                        for kw in range(ker_w):
                            # Calculate the corresponding input coordinates
                            ih = oh * stride + kh - padding
                            iw = ow * stride + kw - padding

                            # Add the weight to the matrix if it's within the valid input bounds
                            if 0 <= ih < input_h and 0 <= iw < input_w:
                                # Calculate the column index for the flattened input vector
                                col_idx = ic * (input_h * input_w) + ih * input_w + iw
                                toeplitz_tensor[row_idx, col_idx] = kernel[
                                    oc, ic, kh, kw
                                ]
    return toeplitz_tensor, output_shape


def orion_conv2d(input_vector, kernel, stride, padding, input_shape):
    in_c, in_h, in_w = input_shape
    out_c, in_c, ker_h, ker_w = kernel.shape

    toeplitz_matrix, output_shape = create_toeplitz_matrix(
        kernel, in_h, in_w, stride, padding
    )
    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    return output_vector, output_shape


def create_multiplexed_toeplitz_for_bsgs(
    kernel, input_h, input_w, stride, padding, input_gap, output_gap
):
    out_c, in_c, ker_h, ker_w = kernel.shape

    # Actual output dimensions
    out_h = (input_h + 2 * padding - ker_h) // stride + 1
    out_w = (input_w + 2 * padding - ker_w) // stride + 1

    # FHE dimensions considering gap
    iG = input_gap
    oG = output_gap

    # FHE input/output shapes
    on_Ci = math.ceil(in_c / (iG**2)) if iG > 1 else in_c
    on_Hi = input_h * iG if iG > 1 else input_h
    on_Wi = input_w * iG if iG > 1 else input_w

    on_Co = math.ceil(out_c / (oG**2))
    on_Ho = max(on_Hi, out_h * oG)
    on_Wo = max(on_Wi, out_w * oG)

    print(
        f"Multiplexed Toeplitz - Input gap: {iG}, Output gap: {oG}, stride: {stride}, padding: {padding}"
    )
    print(
        f"Clear shapes: in=({in_c}, {input_h}, {input_w}), out=({out_c}, {out_h}, {out_w})"
    )
    print(
        f"FHE shapes: in=({on_Ci}, {on_Hi}, {on_Wi}), out=({on_Co}, {on_Ho}, {on_Wo})"
    )

    # Initialize Toeplitz matrix
    n_rows = on_Co * on_Ho * on_Wo
    n_cols = on_Ci * on_Hi * on_Wi
    toeplitz = torch.zeros((n_rows, n_cols), dtype=kernel.dtype)

    # Process each output position
    for ho in range(out_h):
        for wo in range(out_w):
            for oc in range(out_c):
                # Determine multiplexed output position
                if oG > 1:
                    mpx_oc = oc // (oG**2)
                    oc_offset = oc % (oG**2)
                    oc_row = oc_offset // oG
                    oc_col = oc_offset % oG
                    mpx_ho = ho * oG + oc_row
                    mpx_wo = wo * oG + oc_col
                else:
                    mpx_oc = oc
                    mpx_ho = ho
                    mpx_wo = wo

                # Output row index in Toeplitz matrix
                row = mpx_oc * (on_Ho * on_Wo) + mpx_ho * on_Wo + mpx_wo

                # Process each kernel position
                for ic in range(in_c):
                    for kh in range(ker_h):
                        for kw in range(ker_w):
                            # Input position in original space
                            ih = ho * stride + kh - padding
                            iw = wo * stride + kw - padding

                            if 0 <= ih < input_h and 0 <= iw < input_w:
                                # Determine multiplexed input position
                                if iG > 1:
                                    mpx_ic = ic // (iG**2)
                                    ic_offset = ic % (iG**2)
                                    ic_row = ic_offset // iG
                                    ic_col = ic_offset % iG
                                    mpx_ih = ih * iG + ic_row
                                    mpx_iw = iw * iG + ic_col
                                else:
                                    mpx_ic = ic
                                    mpx_ih = ih
                                    mpx_iw = iw

                                # Input column index in Toeplitz matrix
                                col = mpx_ic * (on_Hi * on_Wi) + mpx_ih * on_Wi + mpx_iw

                                if row < n_rows and col < n_cols:
                                    toeplitz[row, col] = kernel[oc, ic, kh, kw]

    return toeplitz, (out_c, out_h, out_w)


def orion_conv2d_with_gap(
    input_vector, kernel, stride, padding, input_shape, input_gap=1, output_gap=None
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride

    print(
        f"\n== Orion Conv2d with gap == Input gap: {input_gap}, Output gap: {output_gap}, Stride: {stride}"
    )
    toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
        kernel, in_h, in_w, stride, padding, input_gap, output_gap
    )
    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    return output_vector, output_shape, output_gap


def orion_conv2d_cached(
    input_vector,
    kernel,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
):
    in_c, in_h, in_w = input_shape
    out_c, in_c, ker_h, ker_w = kernel.shape
    try:
        cache_dir = Path(cache_dir)
    except NameError:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached Toeplitz matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_matrix = cached_data["toeplitz_matrix"]
        output_shape = cached_data["output_shape"]
    else:
        print(f"Computing new Toeplitz matrix: {cache_name}")
        toeplitz_matrix, output_shape = create_toeplitz_matrix(
            kernel, in_h, in_w, stride, padding
        )
        cached_data = {
            "toeplitz_matrix": toeplitz_matrix.cpu(),
            "output_shape": output_shape,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached Toeplitz matrix: {cache_file}")
    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    return output_vector, output_shape


def orion_conv2d_cached_with_gap(
    input_vector,
    kernel,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride
    print(
        f"Multiplexed Toeplitz - Input gap: {input_gap}, Output gap: {output_gap}, stride: {stride}, padding: {padding}"
    )

    try:
        cache_dir = Path(cache_dir)
    except NameError:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached Toeplitz matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_matrix = cached_data["toeplitz_matrix"]
        output_shape = cached_data["output_shape"]
    else:
        print(f"Computing new Toeplitz matrix: {cache_name}")
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            kernel, in_h, in_w, stride, padding, input_gap, output_gap
        )
        cached_data = {
            "toeplitz_matrix": toeplitz_matrix.cpu(),
            "output_shape": output_shape,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached Toeplitz matrix: {cache_file}")
    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    return output_vector, output_shape, output_gap


def orion_bn(input, bn_layer, output_shape, scale_factor=32):
    out_c, out_h, out_w = output_shape
    bn = bn_layer
    W, B, M, V, E = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    G = W / torch.sqrt(V + E)
    H = B - G * M
    num_repeats = out_h * out_w
    G_expanded = torch.repeat_interleave(G, repeats=num_repeats)
    H_expanded = torch.repeat_interleave(H, repeats=num_repeats)
    H_expanded = H_expanded / scale_factor
    G_expanded = F.pad(
        G_expanded, (0, N - G_expanded.shape[0]), mode="constant", value=0
    )
    H_expanded = F.pad(
        H_expanded, (0, N - H_expanded.shape[0]), mode="constant", value=0
    )
    input[0] = input[0] * G_expanded + H_expanded
    return input


def orion_bn_with_gap(input, bn_layer, output_shape, gap, scale_factor=32):
    out_c, out_h, out_w = output_shape
    bn = bn_layer
    W, B, M, V, E = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    G = W / torch.sqrt(V + E)
    H = B - G * M

    if gap == 1:
        num_repeats = out_h * out_w
        G_expanded = torch.repeat_interleave(G, repeats=num_repeats)
        H_expanded = torch.repeat_interleave(H, repeats=num_repeats)
    else:

        def multiplex(matrix, gap):
            N, Ci, Hi, Wi = matrix.shape
            Co = math.ceil(Ci / (gap**2))

            # Pad the tensor to have channels divisible by gap^2
            padded = torch.zeros(N, Co * gap**2, Hi, Wi)
            padded[:, :Ci, ...] = matrix
            return F.pixel_shuffle(padded, gap)  # multiplexed

        # Expand BatchNorm parameters to 4D (batch=1, channels, height, width)
        # Here height, width are actual output sizes
        G_4d = G.view(1, out_c, 1, 1).expand(1, out_c, out_h, out_w)
        H_4d = H.view(1, out_c, 1, 1).expand(1, out_c, out_h, out_w)

        # Apply multiplex
        G_multiplexed = multiplex(G_4d, gap).squeeze(0)  # (mpx_c, mpx_h, mpx_w)
        H_multiplexed = multiplex(H_4d, gap).squeeze(0)

        # Flatten
        G_expanded = G_multiplexed.flatten()
        H_expanded = H_multiplexed.flatten()

    H_expanded = H_expanded / scale_factor
    G_expanded = F.pad(
        G_expanded, (0, N - G_expanded.shape[0]), mode="constant", value=0
    )
    H_expanded = F.pad(
        H_expanded, (0, N - H_expanded.shape[0]), mode="constant", value=0
    )
    input[0] = input[0] * G_expanded + H_expanded
    return input


def create_shortcut_mask(input_shape, output_shape, input_gap, output_gap, stride):
    """
    Create a mask for shortcut connection when dimensions change.
    Used for downsampling and channel projection.
    """
    in_c, in_h, in_w = input_shape
    out_c, out_h, out_w = output_shape

    # Calculate expected output dimensions after stride
    expected_h = (in_h + stride - 1) // stride
    expected_w = (in_w + stride - 1) // stride

    assert (
        expected_h == out_h and expected_w == out_w
    ), f"Dimension mismatch: expected ({expected_h}, {expected_w}), got ({out_h}, {out_w})"

    # Create mask based on gap configuration
    if input_gap == output_gap:
        # Same gap - simple channel/spatial downsampling
        mask = torch.zeros(2**15)

        if output_gap == 1:
            # No multiplexing
            for c in range(min(in_c, out_c)):
                for h in range(out_h):
                    for w in range(out_w):
                        src_h = h * stride
                        src_w = w * stride
                        if src_h < in_h and src_w < in_w:
                            src_idx = c * in_h * in_w + src_h * in_w + src_w
                            dst_idx = c * out_h * out_w + h * out_w + w
                            mask[dst_idx] = 1.0
        else:
            # With multiplexing - need to handle gap properly
            mpx_in_c = math.ceil(in_c / (input_gap**2))
            mpx_in_h = in_h * input_gap
            mpx_in_w = in_w * input_gap

            mpx_out_c = math.ceil(out_c / (output_gap**2))
            mpx_out_h = out_h * output_gap
            mpx_out_w = out_w * output_gap

            for c in range(min(mpx_in_c, mpx_out_c)):
                for h in range(mpx_out_h):
                    for w in range(mpx_out_w):
                        src_h = h * stride
                        src_w = w * stride
                        if src_h < mpx_in_h and src_w < mpx_in_w:
                            src_idx = c * mpx_in_h * mpx_in_w + src_h * mpx_in_w + src_w
                            dst_idx = c * mpx_out_h * mpx_out_w + h * mpx_out_w + w
                            if dst_idx < 2**15:
                                mask[dst_idx] = 1.0
    else:
        # Gap change - need special handling
        mask = torch.zeros(2**15)
        # This case requires demux/remux which is complex
        # For now, we'll use a simplified approach
        print(f"Warning: Gap change in shortcut from {input_gap} to {output_gap}")

    return mask


def orion_shortcut_conv2d_with_gap(
    input_vector,
    shortcut_module,
    input_shape,
    input_gap,
    output_gap,
    cache_name,
    cache_dir,
    is_caching=False,
):
    """
    Apply shortcut connection with 1x1 convolution for dimension matching.
    """
    if isinstance(shortcut_module, nn.Sequential) and len(shortcut_module) == 0:
        # Identity shortcut - no transformation needed
        return input_vector, input_shape, input_gap

    elif (
        hasattr(shortcut_module, "__class__")
        and shortcut_module.__class__.__name__ == "LambdaLayer"
    ):
        # Handle LambdaLayer case - convert to 1x1 conv operation
        # LambdaLayer does: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4))
        # This is equivalent to stride=2 downsampling + channel padding

        in_c, in_h, in_w = input_shape

        # In ResNet, when stride=2, channels typically double (16->32, 32->64)
        # LambdaLayer pads with planes//4 on each side
        # So output channels = input channels * 2
        out_c = in_c * 2
        stride = 2

        # Create a 1x1 conv weight that mimics the LambdaLayer behavior
        # We'll create a conv that maps input channels to output channels with zero padding
        conv_weight = torch.zeros(out_c, in_c, 1, 1).double()

        # Map the existing channels to the middle of output channels
        # LambdaLayer pads with planes//4 on each side
        # For out_c channels, padding is out_c//4 on each side
        start_idx = out_c // 4  # This centers the input channels
        for i in range(in_c):
            conv_weight[start_idx + i, i, 0, 0] = 1.0

        # Apply the 1x1 conv with stride=2
        out_vector, out_shape, current_gap = orion_conv2d_cached_with_gap(
            input_vector,
            conv_weight,
            stride,
            0,
            input_shape,
            cache_name=cache_name,
            cache_dir=cache_dir,
            is_caching=is_caching,
            input_gap=input_gap,
            output_gap=output_gap,
        )

        return out_vector, out_shape, current_gap

    else:
        # Regular Sequential with Conv2d + BatchNorm2d
        conv_layer = shortcut_module[0]
        bn_layer = shortcut_module[1] if len(shortcut_module) > 1 else None

        # Apply 1x1 convolution
        out_vector, out_shape, current_gap = orion_conv2d_cached_with_gap(
            input_vector,
            conv_layer.weight.detach(),
            conv_layer.stride[0],
            conv_layer.padding[0],
            input_shape,
            cache_name=cache_name,
            cache_dir=cache_dir,
            is_caching=is_caching,
            input_gap=input_gap,
            output_gap=output_gap,
        )

        # Apply batch norm if exists
        if bn_layer is not None:
            out_vector = orion_bn_with_gap(out_vector, bn_layer, out_shape, current_gap)

        return out_vector, out_shape, current_gap


def orion_avgpool_global(
    input_vector,
    input_shape,
    input_gap,
    output_gap=32,
    cache_name=None,
    cache_dir=None,
    is_caching=False,
):
    in_c, in_h, in_w = input_shape

    # AdaptiveAvgPool2d with output_size=(1,1) is a grouped convolution
    # Each output channel only connects to the corresponding input channel
    spatial_size = in_h * in_w

    # Create grouped convolution kernel: each group processes one channel
    # kernel shape: (out_channels, in_channels/groups, kernel_h, kernel_w)
    # For grouped conv with groups=in_c: weight shape is (in_c, 1, in_h, in_w)
    # But we need to pass the full weight as if it's ungrouped for Toeplitz construction
    avg_weight = torch.zeros(in_c, in_c, in_h, in_w, dtype=torch.double)

    # Set up as diagonal - each output channel only sees its corresponding input channel
    for c in range(in_c):
        avg_weight[c, c, :, :] = 1.0 / spatial_size

    # For global average pooling: stride = input size, no padding
    stride = in_h  # This should be 8 for 8x8 input
    padding = 0

    # IMPORTANT: For AdaptiveAvgPool2d, the gap should RESET to 1
    # because we're reducing spatial dimensions to 1x1
    # This is consistent with Orion's pooling implementation
    result, out_shape, output_gap = orion_avgpool_global_cached(
        input_vector,
        avg_weight,
        stride,
        padding,
        input_shape,
        cache_name=cache_name,
        cache_dir=cache_dir,
        is_caching=False,
        input_gap=input_gap,
        output_gap=output_gap,  # Gap resets to 1 for global pooling
    )

    # Output shape should be (C, 1, 1)
    output_shape = (in_c, 1, 1)
    return result, output_shape, output_gap  # Return output_gap=1


def orion_avgpool_global_cached(
    input_vector,
    kernel,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride
    print(
        f"Multiplexed Toeplitz - Input gap: {input_gap}, Output gap: {output_gap}, stride: {stride}, padding: {padding}"
    )

    try:
        cache_dir = Path(cache_dir)
    except NameError:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached Toeplitz matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_matrix = cached_data["toeplitz_matrix"]
        output_shape = cached_data["output_shape"]
    else:
        print(f"Computing new Toeplitz matrix: {cache_name}")
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            kernel, in_h, in_w, stride, padding, input_gap, output_gap
        )
        cached_data = {
            "toeplitz_matrix": toeplitz_matrix.cpu(),
            "output_shape": output_shape,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached Toeplitz matrix: {cache_file}")
    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    return output_vector, output_shape, output_gap


# Regular AvgPool2d with gap support
def orion_avgpool2d_with_gap(
    input_vector,
    kernel_size,
    stride,
    padding,
    input_shape,
    input_gap=1,
    output_gap=None,
    cache_name=None,
    cache_dir=None,
    is_caching=False,
):
    """
    Regular average pooling with gap support for AlexNet features.1 and features.3
    """
    in_c, in_h, in_w = input_shape

    # Calculate output gap based on stride if not specified
    if output_gap is None:
        output_gap = input_gap * stride

    # Create average pooling kernel
    # For regular AvgPool2d, we apply the same kernel to all channels
    kernel = torch.ones(1, 1, kernel_size, kernel_size, dtype=torch.double) / (
        kernel_size * kernel_size
    )

    # Expand kernel to all channels (but still as ungrouped for Toeplitz)
    avg_weight = torch.zeros(in_c, in_c, kernel_size, kernel_size, dtype=torch.double)
    for c in range(in_c):
        avg_weight[c, c, :, :] = 1.0 / (kernel_size * kernel_size)

    print(
        f"AvgPool2d - Input shape: {input_shape}, kernel_size: {kernel_size}, stride: {stride}"
    )
    print(f"Input gap: {input_gap} -> Output gap: {output_gap}")

    result, out_shape, final_gap = orion_avgpool_cached_with_gap(
        input_vector,
        avg_weight,
        stride,
        padding,
        input_shape,
        cache_name=cache_name,
        cache_dir=cache_dir,
        is_caching=is_caching,
        input_gap=input_gap,
        output_gap=output_gap,
    )

    return result, out_shape, final_gap


def orion_avgpool_cached_with_gap(
    input_vector,
    kernel,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
):
    in_c, in_h, in_w = input_shape

    if output_gap is None:
        output_gap = input_gap * stride

    print(
        f"Multiplexed AvgPool Toeplitz - Input gap: {input_gap}, Output gap: {output_gap}"
    )

    try:
        cache_dir = Path(cache_dir)
    except:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}_g{input_gap}to{output_gap}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached Toeplitz matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_matrix = cached_data["toeplitz_matrix"]
        output_shape = cached_data["output_shape"]
    else:
        print(f"Computing new Toeplitz matrix: {cache_name}")
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            kernel, in_h, in_w, stride, padding, input_gap, output_gap
        )
        cached_data = {
            "toeplitz_matrix": toeplitz_matrix.cpu(),
            "output_shape": output_shape,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached Toeplitz matrix: {cache_file}")

    output_vector = conv_gemv_cp(input_vector, toeplitz_matrix, N)

    # Calculate actual output shape
    out_h = (in_h + 2 * padding - kernel.shape[2]) // stride + 1
    out_w = (in_w + 2 * padding - kernel.shape[3]) // stride + 1
    output_shape = (in_c, out_h, out_w)

    return output_vector, output_shape, output_gap


# ------------------------------------------------------------------
# Multi-Cipher Conv2d Functions
# ------------------------------------------------------------------
def orion_conv2d_cached_with_gap_multi_cipher(
    input_vector,
    kernel,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
    num_input_ciphers=1,
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride

    print(
        f"\n== Multi-Cipher Conv2d == Input gap: {input_gap}, Output gap: {output_gap}, Stride: {stride}, # of input ciphers: {num_input_ciphers}"
    )

    # Cache handling
    try:
        cache_dir = Path(cache_dir)
    except NameError:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}_multi.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached multi-cipher Toeplitz matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_blocks = cached_data["toeplitz_blocks"]
        output_shape = cached_data["output_shape"]
        num_row_blocks = cached_data["num_row_blocks"]
        num_col_blocks = cached_data["num_col_blocks"]
        print(
            f"Output shape: {output_shape}, Block sizes: ({num_row_blocks}, {num_col_blocks})"
        )
        for i in range(num_row_blocks):
            for j in range(num_col_blocks):
                print(f"Block {i},{j}: {toeplitz_blocks[i][j].shape}")
    else:
        print(f"Computing new multi-cipher Toeplitz matrix: {cache_name}")
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            kernel, in_h, in_w, stride, padding, input_gap, output_gap
        )
        num_rows, num_cols = toeplitz_matrix.shape
        num_row_blocks = math.ceil(num_rows / N)
        num_col_blocks = math.ceil(num_cols / N)

        print(
            f"Toeplitz shape: ({num_rows}, {num_cols}), num_row_blocks: {num_row_blocks}, num_col_blocks: {num_col_blocks}"
        )

        # If single block, no need to split
        if num_row_blocks == 1 and num_col_blocks == 1:
            toeplitz_blocks = [[toeplitz_matrix]]
        else:
            # Split into blocks
            toeplitz_blocks = []
            for i in range(num_row_blocks):
                row_blocks = []
                for j in range(num_col_blocks):
                    # Extract block
                    row_start = i * N
                    row_end = min((i + 1) * N, num_rows)
                    col_start = j * N
                    col_end = min((j + 1) * N, num_cols)

                    block = toeplitz_matrix[row_start:row_end, col_start:col_end]

                    #    # Pad block to N x N if needed
                    #    if block.shape[0] < N or block.shape[1] < N:
                    #        padded_block = torch.zeros(N, N, dtype=block.dtype)
                    #        padded_block[:block.shape[0], :block.shape[1]] = block
                    #        block = padded_block

                    row_blocks.append(block)
                toeplitz_blocks.append(row_blocks)

        # Cache the blocks
        cached_data = {
            "toeplitz_blocks": toeplitz_blocks,
            "output_shape": output_shape,
            "num_row_blocks": num_row_blocks,
            "num_col_blocks": num_col_blocks,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached multi-cipher Toeplitz matrix: {cache_file}")

    # If single block in both dimensions, use original function
    if num_row_blocks == 1 and num_col_blocks == 1:
        print("Using single-cipher convolution (matrix fits in one block)")
        # Ensure input is single cipher format
        if num_input_ciphers > 1:
            # Combine multi-cipher input into single cipher if possible
            combined_input = np.full((1,), Empty(), dtype=object)
            combined_input[0] = input_vector[0]
            for i in range(1, num_input_ciphers):
                combined_input[0] = combined_input[0] + roll(input_vector[i], i * N)
            out, out_shape, output_gap = orion_conv2d_cached_with_gap(
                combined_input,
                kernel,
                stride,
                padding,
                input_shape,
                cache_name,
                cache_dir,
                True,
                input_gap,
                output_gap,
            )
            return out, out_shape, output_gap, 1
        else:
            out, out_shape, output_gap = orion_conv2d_cached_with_gap(
                input_vector,
                kernel,
                stride,
                padding,
                input_shape,
                cache_name,
                cache_dir,
                True,
                input_gap,
                output_gap,
            )
            return out, out_shape, output_gap, 1

    print(
        f"== Using multi-cipher convolution == Blocks: ({num_row_blocks}, {num_col_blocks})"
    )
    input_blocks = np.full((num_col_blocks,), Empty(), dtype=object)
    if num_input_ciphers == 1:
        if num_input_ciphers != num_col_blocks:
            # Safe check
            raise ValueError(
                f"Input cipher count ({num_input_ciphers}) doesn't match column blocks ({num_col_blocks})"
            )
        input_blocks[0] = input_vector[0]
    else:
        if num_input_ciphers != num_col_blocks:
            # Safe check
            raise ValueError(
                f"Input cipher count ({num_input_ciphers}) doesn't match column blocks ({num_col_blocks})"
            )
        for j in range(num_col_blocks):
            input_blocks[j] = input_vector[j]

    result = np.full((num_row_blocks,), Empty(), dtype=object)
    partial = np.full((1,), Empty(), dtype=object)
    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            print(f"\n== Block {i},{j} == {toeplitz_blocks[i][j].shape}")
            partial[0] = input_blocks[j]
            partial = conv_gemv_cp(partial, toeplitz_blocks[i][j], N)
            if isinstance(result[i], Empty):
                result[i] = partial[0]
            else:
                result[i] = result[i] + partial[0]

    out_c, out_h, out_w = output_shape
    total_output_size = out_c * out_h * out_w

    if total_output_size <= N and num_row_blocks > 1:
        print(f"Combining {num_row_blocks} output blocks into single cipher")
        combined = np.full((1,), Empty(), dtype=object)
        combined[0] = result[0]

        for i in range(1, num_row_blocks):
            shifted = roll(result[i], i * N)
            combined[0] = combined[0] + shifted

        return combined, output_shape, output_gap, 1
    else:
        print(f"Returning {num_row_blocks} output ciphers")
        return result, output_shape, output_gap, num_row_blocks


def orion_bn_with_gap_multi_cipher(
    input, bn_layer, output_shape, gap, num_ciphers, scale_factor=32
):
    if num_ciphers == 1:
        return orion_bn_with_gap(input, bn_layer, output_shape, gap, scale_factor)

    out_c, out_h, out_w = output_shape
    bn = bn_layer
    W, B, M, V, E = bn.weight, bn.bias, bn.running_mean, bn.running_var, bn.eps
    G = W / torch.sqrt(V + E)
    H = B - G * M

    if gap == 1:
        num_repeats = out_h * out_w
        G_expanded = torch.repeat_interleave(G, repeats=num_repeats)
        H_expanded = torch.repeat_interleave(H, repeats=num_repeats)
    else:

        def multiplex(matrix, gap):
            N, Ci, Hi, Wi = matrix.shape
            Co = math.ceil(Ci / (gap**2))

            # Pad the tensor to have channels divisible by gap^2
            padded = torch.zeros(N, Co * gap**2, Hi, Wi)
            padded[:, :Ci, ...] = matrix
            return F.pixel_shuffle(padded, gap)  # multiplexed

        # Expand BatchNorm parameters to 4D (batch=1, channels, height, width)
        # Here height, width are actual output sizes
        G_4d = G.view(1, out_c, 1, 1).expand(1, out_c, out_h, out_w)
        H_4d = H.view(1, out_c, 1, 1).expand(1, out_c, out_h, out_w)

        # Apply multiplex
        G_multiplexed = multiplex(G_4d, gap).squeeze(0)  # (mpx_c, mpx_h, mpx_w)
        H_multiplexed = multiplex(H_4d, gap).squeeze(0)

        # Flatten
        G_expanded = G_multiplexed.flatten()
        H_expanded = H_multiplexed.flatten()

    H_expanded = H_expanded / scale_factor
    total_mpx_size = G_expanded.shape[0]
    # G_expanded = F.pad(G_expanded, (0, N - G_expanded.shape[0]), mode='constant', value=0)
    # H_expanded = F.pad(H_expanded, (0, N - H_expanded.shape[0]), mode='constant', value=0)
    # input[0] = input[0] * G_expanded + H_expanded

    for i in range(num_ciphers):
        start_index = i * slot_length
        end_index = min((i + 1) * slot_length, total_mpx_size)
        actual_size = end_index - start_index
        if actual_size > 0:
            G_cipher = torch.zeros(slot_length, dtype=G_expanded.dtype)
            H_cipher = torch.zeros(slot_length, dtype=H_expanded.dtype)
            G_cipher[:actual_size] = G_expanded[start_index:end_index]
            H_cipher[:actual_size] = H_expanded[start_index:end_index]
            input[i] = input[i] * G_cipher + H_cipher

    return input


def orion_avgpool2d_cached_with_gap_multi_cipher(
    input_vector,
    kernel_size,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
    num_input_ciphers=1,
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride

    print(
        f"\n== Multi-Cipher AvgPool2d == Kernel: {kernel_size}x{kernel_size}, Stride: {stride}"
    )
    print(
        f"Input gap: {input_gap}, Output gap: {output_gap}, Num input ciphers: {num_input_ciphers}"
    )

    # Create pooling kernel
    avg_weight = torch.zeros(in_c, in_c, kernel_size, kernel_size, dtype=torch.double)
    for c in range(in_c):
        avg_weight[c, c, :, :] = 1.0 / (kernel_size * kernel_size)

    # Cache handling
    try:
        cache_dir = Path(cache_dir)
    except:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}_multi_g{input_gap}to{output_gap}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached multi-cipher pooling matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_blocks = cached_data["toeplitz_blocks"]
        output_shape = cached_data["output_shape"]
        num_row_blocks = cached_data["num_row_blocks"]
        num_col_blocks = cached_data["num_col_blocks"]
    else:
        print(f"Computing new multi-cipher pooling matrix: {cache_name}")

        # Generate full Toeplitz matrix for pooling
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            avg_weight, in_h, in_w, stride, padding, input_gap, output_gap
        )

        # Analyze matrix size
        num_rows, num_cols = toeplitz_matrix.shape
        num_row_blocks = math.ceil(num_rows / N)
        num_col_blocks = math.ceil(num_cols / N)

        print(f"Pooling matrix shape: ({num_rows}, {num_cols})")
        print(f"Matrix blocks: ({num_row_blocks}, {num_col_blocks})")

        # Split into blocks if needed
        if num_row_blocks == 1 and num_col_blocks == 1:
            toeplitz_blocks = [[toeplitz_matrix]]
        else:
            toeplitz_blocks = []
            for i in range(num_row_blocks):
                row_blocks = []
                for j in range(num_col_blocks):
                    row_start = i * N
                    row_end = min((i + 1) * N, num_rows)
                    col_start = j * N
                    col_end = min((j + 1) * N, num_cols)

                    block = toeplitz_matrix[row_start:row_end, col_start:col_end]
                    row_blocks.append(block)
                toeplitz_blocks.append(row_blocks)

        # Cache the blocks
        cached_data = {
            "toeplitz_blocks": toeplitz_blocks,
            "output_shape": output_shape,
            "num_row_blocks": num_row_blocks,
            "num_col_blocks": num_col_blocks,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached multi-cipher pooling matrix: {cache_file}")

    # Process based on block configuration
    if num_row_blocks == 1 and num_col_blocks == 1:
        # Single block - use regular function
        print("Using single-cipher pooling (matrix fits in one block)")
        if num_input_ciphers > 1:
            # Combine multi-cipher input
            combined_input = np.full((1,), Empty(), dtype=object)
            combined_input[0] = input_vector[0]
            for i in range(1, num_input_ciphers):
                combined_input[0] = combined_input[0] + roll(input_vector[i], i * N)
            single_result = conv_gemv_cp(combined_input, toeplitz_blocks[0][0], N)
            return single_result, output_shape, output_gap, 1
        else:
            single_result = conv_gemv_cp(input_vector, toeplitz_blocks[0][0], N)
            return single_result, output_shape, output_gap, 1

    # Multi-block processing
    print(
        f"Using multi-cipher pooling computation == Blocks: ({num_row_blocks}, {num_col_blocks})"
    )

    # Prepare input blocks
    input_blocks = np.full((num_col_blocks,), Empty(), dtype=object)
    if num_input_ciphers == 1 and num_col_blocks > 1:
        # Need to split single cipher into blocks
        for j in range(num_col_blocks):
            input_blocks[j] = input_vector[0]
    elif num_input_ciphers == num_col_blocks:
        for j in range(num_col_blocks):
            input_blocks[j] = input_vector[j]
    else:
        raise ValueError(
            f"Input cipher count ({num_input_ciphers}) doesn't match column blocks ({num_col_blocks})"
        )

    # Compute output blocks
    result = np.full((num_row_blocks,), Empty(), dtype=object)
    partial = np.full((1,), Empty(), dtype=object)

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            print(f"\n== Block {i},{j} == {toeplitz_blocks[i][j].shape}")
            partial[0] = input_blocks[j]
            partial = conv_gemv_cp(partial, toeplitz_blocks[i][j], N)

            if isinstance(result[i], Empty):
                result[i] = partial[0]
            else:
                result[i] = result[i] + partial[0]

    # Check if we can combine output
    out_c, out_h, out_w = output_shape
    total_output_size = out_c * out_h * out_w

    if total_output_size <= N and num_row_blocks > 1:
        print(f"Combining {num_row_blocks} output blocks into single cipher")
        combined = np.full((1,), Empty(), dtype=object)
        combined[0] = result[0]

        for i in range(1, num_row_blocks):
            shifted = roll(result[i], i * N)
            combined[0] = combined[0] + shifted

        return combined, output_shape, output_gap, 1
    else:
        print(f"Returning {num_row_blocks} output ciphers")
        return result, output_shape, output_gap, num_row_blocks


def orion_avgpool2d_preboot(
    input_vector,
    kernel_size,
    stride,
    padding,
    input_shape,
    cache_name,
    cache_dir,
    is_caching=False,
    input_gap=1,
    output_gap=None,
    num_input_ciphers=1,
):
    in_c, in_h, in_w = input_shape
    if output_gap is None:
        output_gap = input_gap * stride

    print(
        f"\n== Multi-Cipher AvgPool2d == Kernel: {kernel_size}x{kernel_size}, Stride: {stride}"
    )
    print(
        f"Input gap: {input_gap}, Output gap: {output_gap}, Num input ciphers: {num_input_ciphers}"
    )

    # Create pooling kernel
    avg_weight = torch.zeros(in_c, in_c, kernel_size, kernel_size, dtype=torch.double)
    for c in range(in_c):
        avg_weight[c, c, :, :] = 1.0 / (kernel_size * kernel_size * (1 / 32))

    # Cache handling
    try:
        cache_dir = Path(cache_dir)
    except:
        cache_dir = Path(os.getcwd()) / f"../../data/{cache_dir}"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{cache_name}_multi_g{input_gap}to{output_gap}.pt"

    if cache_file.exists() and is_caching:
        print(f"Loading cached multi-cipher pooling matrix: {cache_file}")
        cached_data = torch.load(cache_file, map_location="cpu")
        toeplitz_blocks = cached_data["toeplitz_blocks"]
        output_shape = cached_data["output_shape"]
        num_row_blocks = cached_data["num_row_blocks"]
        num_col_blocks = cached_data["num_col_blocks"]
    else:
        print(f"Computing new multi-cipher pooling matrix: {cache_name}")

        # Generate full Toeplitz matrix for pooling
        toeplitz_matrix, output_shape = create_multiplexed_toeplitz_for_bsgs(
            avg_weight, in_h, in_w, stride, padding, input_gap, output_gap
        )

        # Analyze matrix size
        num_rows, num_cols = toeplitz_matrix.shape
        num_row_blocks = math.ceil(num_rows / N)
        num_col_blocks = math.ceil(num_cols / N)

        print(f"Pooling matrix shape: ({num_rows}, {num_cols})")
        print(f"Matrix blocks: ({num_row_blocks}, {num_col_blocks})")

        # Split into blocks if needed
        if num_row_blocks == 1 and num_col_blocks == 1:
            toeplitz_blocks = [[toeplitz_matrix]]
        else:
            toeplitz_blocks = []
            for i in range(num_row_blocks):
                row_blocks = []
                for j in range(num_col_blocks):
                    row_start = i * N
                    row_end = min((i + 1) * N, num_rows)
                    col_start = j * N
                    col_end = min((j + 1) * N, num_cols)

                    block = toeplitz_matrix[row_start:row_end, col_start:col_end]
                    row_blocks.append(block)
                toeplitz_blocks.append(row_blocks)

        # Cache the blocks
        cached_data = {
            "toeplitz_blocks": toeplitz_blocks,
            "output_shape": output_shape,
            "num_row_blocks": num_row_blocks,
            "num_col_blocks": num_col_blocks,
        }
        torch.save(cached_data, cache_file)
        print(f"Cached multi-cipher pooling matrix: {cache_file}")

    # Process based on block configuration
    if num_row_blocks == 1 and num_col_blocks == 1:
        # Single block - use regular function
        print("Using single-cipher pooling (matrix fits in one block)")
        if num_input_ciphers > 1:
            # Combine multi-cipher input
            combined_input = np.full((1,), Empty(), dtype=object)
            combined_input[0] = input_vector[0]
            for i in range(1, num_input_ciphers):
                combined_input[0] = combined_input[0] + roll(input_vector[i], i * N)
            single_result = conv_gemv_cp(combined_input, toeplitz_blocks[0][0], N)
            return single_result, output_shape, output_gap, 1
        else:
            single_result = conv_gemv_cp(input_vector, toeplitz_blocks[0][0], N)
            return single_result, output_shape, output_gap, 1

    # Multi-block processing
    print(
        f"Using multi-cipher pooling computation == Blocks: ({num_row_blocks}, {num_col_blocks})"
    )

    # Prepare input blocks
    input_blocks = np.full((num_col_blocks,), Empty(), dtype=object)
    if num_input_ciphers == 1 and num_col_blocks > 1:
        # Need to split single cipher into blocks
        for j in range(num_col_blocks):
            input_blocks[j] = input_vector[0]
    elif num_input_ciphers == num_col_blocks:
        for j in range(num_col_blocks):
            input_blocks[j] = input_vector[j]
    else:
        raise ValueError(
            f"Input cipher count ({num_input_ciphers}) doesn't match column blocks ({num_col_blocks})"
        )

    # Compute output blocks
    result = np.full((num_row_blocks,), Empty(), dtype=object)
    partial = np.full((1,), Empty(), dtype=object)

    for i in range(num_row_blocks):
        for j in range(num_col_blocks):
            print(f"\n== Block {i},{j} == {toeplitz_blocks[i][j].shape}")
            partial[0] = input_blocks[j]
            partial = conv_gemv_cp(partial, toeplitz_blocks[i][j], N)

            if isinstance(result[i], Empty):
                result[i] = partial[0]
            else:
                result[i] = result[i] + partial[0]

    # Check if we can combine output
    out_c, out_h, out_w = output_shape
    total_output_size = out_c * out_h * out_w

    if total_output_size <= N and num_row_blocks > 1:
        print(f"Combining {num_row_blocks} output blocks into single cipher")
        combined = np.full((1,), Empty(), dtype=object)
        combined[0] = result[0]

        for i in range(1, num_row_blocks):
            shifted = roll(result[i], i * N)
            combined[0] = combined[0] + shifted

        return combined, output_shape, output_gap, 1
    else:
        print(f"Returning {num_row_blocks} output ciphers")
        return result, output_shape, output_gap, num_row_blocks


def prepare_fc_weight_for_multiplexed_input(fc_weight, input_shape, input_gap):
    out_dim, in_features = fc_weight.shape
    in_c, in_h, in_w = input_shape

    assert (
        in_features == in_c * in_h * in_w
    ), f"Dimension mismatch: {in_features} != {in_c * in_h * in_w}"

    if input_gap == 1:
        return fc_weight

    # Calculate multiplexed layout
    mpx_c = math.ceil(in_c / (input_gap**2))
    mpx_h = in_h * input_gap
    mpx_w = in_w * input_gap

    # Create rearranged weight matrix
    rearranged = torch.zeros_like(fc_weight)

    for out_idx in range(out_dim):
        for c in range(in_c):
            for h in range(in_h):
                for w in range(in_w):
                    # Original weight position
                    orig_idx = c * (in_h * in_w) + h * in_w + w

                    # Multiplexed position
                    mpx_channel = c // (input_gap**2)
                    c_offset = c % (input_gap**2)
                    c_row = c_offset // input_gap
                    c_col = c_offset % input_gap

                    mpx_h_pos = h * input_gap + c_row
                    mpx_w_pos = w * input_gap + c_col
                    mpx_idx = (
                        mpx_channel * (mpx_h * mpx_w) + mpx_h_pos * mpx_w + mpx_w_pos
                    )

                    # Copy weight to new position
                    if mpx_idx < in_features:
                        rearranged[out_idx, mpx_idx] = fc_weight[out_idx, orig_idx]

    return rearranged


# ------------------------------------------------------------
# Orion Preprocess and Postprocess
# ------------------------------------------------------------
def orion_preprocess(input, scale_factor=32):
    print("input.shape: ", input.shape)
    input_flattened = input.flatten()
    input_flattened = input_flattened / scale_factor
    print("input -> input_flattened: ", input.shape, "->", input_flattened.shape)
    input_flattened = F.pad(
        input_flattened, (0, 2**15 - input_flattened.shape[0]), mode="constant", value=0
    )
    return input_flattened


def postprocess(flattened_output, output_shape, scale_factor=32):
    torch_res_size = 1
    for i in range(len(output_shape)):
        torch_res_size *= output_shape[i]
    return flattened_output[0, :torch_res_size].reshape(output_shape) * scale_factor


def orion_postprocess(flattened_output, output_shape, scale_factor=32):
    if len(flattened_output.shape) == 2 and flattened_output.shape[0] == 1:
        output = flattened_output[0]
    else:
        output = flattened_output
    output_length = 1
    for i in range(len(output_shape)):
        output_length *= output_shape[i]
    output = output[:output_length]
    result_tensor = output.reshape(output_shape)
    return result_tensor * scale_factor


def orion_multiplexed_postprocess(
    result_tensor, expected_shape, gap=1, scale_factor=32
):
    if gap == 1:
        # No demultiplexing needed
        flat = result_tensor.flatten()
        required = np.prod(expected_shape)
        return flat[:required].reshape(expected_shape) * scale_factor

    # Demultiplexing for gap > 1
    N, out_c, out_h, out_w = expected_shape

    # Calculate multiplexed dimensions
    mpx_c = math.ceil(out_c / (gap**2))
    mpx_h = out_h * gap
    mpx_w = out_w * gap

    # Reshape to multiplexed form
    flat = result_tensor.flatten()
    required = mpx_c * mpx_h * mpx_w
    mpx_tensor = flat[:required].reshape(1, mpx_c, mpx_h, mpx_w)

    # Extract relevant region and demultiplex
    extracted = mpx_tensor[:, :mpx_c, : out_h * gap, : out_w * gap]
    demuxed = F.pixel_unshuffle(extracted, gap)

    # Extract final channels
    result = demuxed[:, :out_c, :out_h, :out_w]
    return result * scale_factor
