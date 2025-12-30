#!/usr/bin/env python
import numpy as np
import math

def roll(input, i) :
    return input.rotate(-i) # rotate to lower index

def cint (i) : 
    return int(np.ceil(i))

def fint (i) : 
    return int(np.floor(i))

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

# Utility Functions for hetorch
def masking_ciphertext(ctxt, input_length, slot_length) :
    mask_input = np.zeros(slot_length, dtype=np.double)
    mask_input[:input_length].fill(1)
    ctxt[0] = ctxt[0] * mask_input
    return ctxt

def full_replicate_ciphertext(ctxt, input_length, slot_length) :
    closest_power_of_two = 2 ** cint(np.log2(input_length))
    print(f"input_length: {input_length}, closest_power_of_two: {closest_power_of_two}")
    # packing_times = slot_length // closest_power_of_two
    # result, packing_size, packing_times = replicate_ciphertext(ctxt, closest_power_of_two, packing_times)
    packing_times = slot_length // input_length
    result, packing_size, packing_times = replicate_ciphertext(ctxt, input_length, packing_times)
    return result, packing_size, packing_times

def replicate_ciphertext(result, input_length, repeat_times) :
    res_power_of_2 = repeat_times - 2 ** fint(np.log2(repeat_times))
    if res_power_of_2 == 1:
        tmp = np.full((1,) ,Empty(), dtype = object) # shape (1,)
        tmp[0] = result[0]
        for i in range(fint(np.log2(repeat_times))) :
            shift = input_length * (2 ** i)
            result[0] = result[0] + result[0].rotate(-shift)
        result[0] = result[0] + tmp[0].rotate(-input_length * (repeat_times - 1))
    else:
        res3_power_of_2 = (repeat_times/3) - 2 ** fint(np.log2(repeat_times/3))
        if res3_power_of_2 == 0: # repeat_times = 3 * (2 ** k)
            result[0] = result[0] + result[0].rotate(input_length) + result[0].rotate(2*input_length)
            shift = input_length * 3
            for i in range(fint(np.log2(repeat_times/3))) :
                shift = shift * 2
                result[0] = result[0] + result[0].rotate(-shift)
        else:
            for i in range(cint(np.log2(repeat_times))) :
                shift = input_length * (2 ** i)
                result[0] = result[0] + result[0].rotate(-shift)
    
    return result, input_length, repeat_times

def print_shift_mapping(shift_mapping):
    if not shift_mapping:
        return
        
    gs_groups = {}    
    for shift, (gs, bs) in shift_mapping.items():
        if gs not in gs_groups:
            gs_groups[gs] = {}
        gs_groups[gs][bs] = shift
    
    all_baby_steps = set()
    for gs_group in gs_groups.values():
        all_baby_steps.update(gs_group.keys())
    
    used_baby_steps = sorted(all_baby_steps)
    
    all_values = list(shift_mapping.keys()) + list(gs_groups.keys())
    max_width = max(len(str(val)) for val in all_values) if all_values else 5
    max_width = max(max_width, 5)
    
    print("GS\\BS  ", end="")
    for bs in used_baby_steps:
        print(f"{bs:<{max_width+1}}", end="")
    print()
    
    print("-" * (max_width + (max_width+2) * len(used_baby_steps)))
    
    for gs_value in sorted(gs_groups.keys()):
        print(f"{gs_value:<{max_width+2}}", end="")
        
        for bs in used_baby_steps:
            if bs in gs_groups[gs_value]:
                shift_value = gs_groups[gs_value][bs]
                print(f"{shift_value:<{max_width+1}}", end="")
            else:
                print(f"{'-':<{max_width+1}}", end="")
        print()

# ------------------------------------------------------------------
# bsgs reference
# ------------------------------------------------------------------
# TODO, move rtp reference bsgs to hetorch.blas from hetorch.blas_r2p
# packed_orion_cal_bsgs_list from r2p

# not used, just for reference
def bsgsCP_no_reshape(input, weight, slot_length, bsgs_ratio=hetorch_bsgs_ratio) :
    out_dim, inter_dim = weight.shape
    print(f"bsgsCP weight.shape: {weight.shape}")
    print(f"bsgsCP bsgs_ratio: {bsgs_ratio}")
    mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    input[0] = mask_input * input[0] # shape (slot_length,)
    
    is_out_dim_bigger = out_dim > inter_dim
    if is_out_dim_bigger :
        expand_inter = cint(out_dim / inter_dim) + 1
        input, _, _ = replicate_ciphertext(input, inter_dim, expand_inter)
        # input = masking_ciphertext(input, expand_inter_dim, slot_length)
        weight = torch.stack([ torch.roll(weight[i,:], -(i%inter_dim)) for i in range(out_dim) ]) # diagonal, same dimension (out_dim, inter_dim)
    else :
        input[0] = input[0] + roll(input[0], inter_dim) # shape (slot_length,)
        weight = torch.stack([ torch.roll(weight[i,:], -i) for i in range(out_dim) ]) # diagonal, (out_dim, inter_dim)
    
    expand = cint(inter_dim / out_dim)
    if is_out_dim_bigger :
        weight = weight.T
    else :
        weight = F.pad(weight, (0, expand*out_dim - inter_dim), mode = 'constant', value = 0) # (out_dim, expand*out_dim)
        weight = einops.rearrange(weight, "i1 (i2 i3) -> i3 (i2 i1)", i1 = out_dim, i2 = expand, i3 = out_dim) # transpose
        print(f"after einops.rearrange, expand, weight.shape: {expand}, {weight.shape}")
    
    giant_step = cint(np.sqrt(out_dim / bsgs_ratio))
    baby_step = cint(out_dim / giant_step)
    print(f"out_dim: {out_dim}, baby_step: {baby_step}, giant_step: {giant_step}")
    
    padded_weight = torch.zeros((out_dim if not is_out_dim_bigger else inter_dim, slot_length), dtype=torch.double)
    for i in range(out_dim) :
        offset = (i // baby_step) * baby_step
        padded_weight[i, offset:offset+expand*out_dim] = weight[i, :]
    
    # inplace operation
    group_offset = 0
    tmp = np.full((1,) ,Empty(), dtype = object) # shape (1,)
    result = np.full((1,) ,Empty(), dtype = object) # shape (1,)
    for i in range(giant_step) :
        offset = baby_step * i
        if torch.any(padded_weight[offset:offset+baby_step, :] != 0) :
            for j in range(baby_step) :
                shift = offset + j
                if shift >= out_dim :
                    break
                if i != 0 and j == 0 :
                    if not isinstance(result[0], Empty) and (offset - group_offset) != 0:
                        result[0] = roll(result[0], (offset - group_offset))
                group_offset = offset
                if torch.any(padded_weight[shift, :] != 0):
                    if i == 0 and j == 0 :
                        tmp[0] = input[0] * padded_weight[shift, :]
                    else :
                        tmp[0] = tmp[0] + roll(input[0], -j) * padded_weight[shift, :]
            if not isinstance(tmp[0], Empty) :
                if isinstance(result[0], Empty) :
                    result[0] = tmp[0]
                else :
                    result[0] = result[0] + tmp[0]
                tmp[0] = Empty() # reset tmp
        else :
            print(f"padded_weight[{offset}:{offset+baby_step}, :] is empty")
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)
    
    # self-sum
    for j in range(cint(np.log2(expand))) :
        result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result

def bsgsCP_with_zeros(input, weight, slot_length) :
    out_dim, inter_dim = weight.shape
    input = masking_ciphertext(input, inter_dim, slot_length)
    
    # replicate ciphertext
    input, packing_size, packing_times = replicate_ciphertext(input, inter_dim, 2)
    baby_step = cint(np.sqrt(inter_dim))
    giant_step = cint(inter_dim / baby_step)
    print(f"bsgsCP_with_zeros inter_dim: {inter_dim}, baby_step: {baby_step}, giant_step: {giant_step}")
    
    weight = torch.stack([ torch.roll(weight[i,:], -i) for i in range(out_dim) ])
    weight = weight.T
    padded_weight = torch.zeros((inter_dim, baby_step * (giant_step-1) + out_dim), dtype=torch.double)
    for i in range(inter_dim) :
        shift = i // baby_step
        shift = shift * baby_step
        padded_weight[i, shift:shift+out_dim] = weight[i, :]
    padded_weight = F.pad(padded_weight, (0, slot_length - padded_weight.shape[1]), mode = 'constant', value = 0)
       
    # inplace operation
    group_offset = 0
    tmp = np.full((1,) ,Empty(), dtype = object) # shape (1,)
    result = np.full((1,) ,Empty(), dtype = object) # shape (1,)
    for i in range(giant_step) :
        shift = baby_step * i
        for j in range(baby_step) :
            if i != 0 and j == 0 :
                if not isinstance(result[0], Empty) and (shift - group_offset) != 0:
                    result[0] = roll(result[0], (shift - group_offset))
            group_offset = shift
            if shift + j < inter_dim :
                if j == 0 :
                    tmp[0] = input[0] * padded_weight[shift + j, :]
                else :
                    tmp[0] = tmp[0] + roll(input[0], -j)* padded_weight[shift + j, :]
        if not isinstance(tmp[0], Empty) :
            if isinstance(result[0], Empty) :
                result[0] = tmp[0]
            else :
                result[0] = result[0] + tmp[0]
            tmp[0] = Empty() # reset tmp
    if group_offset != 0:
        result[0] = roll(result[0], -group_offset)

    return result

# ------------------------------------------------------------------
# MPCB Functions
# ------------------------------------------------------------------
# linear = weight * input + bias
def linear(input, weight, bias, nt) :    
    # input padding and scaling
    result = np.full((1) ,Empty(), dtype = object) # shape (1,)
    outdim, indim = weight.shape
    S = torch.ones(indim) # shape (indim,)
    S = F.pad(S, (0, nt- S.shape[0]), mode= 'constant', value=0) # shape (nt,)
    input[0] = S * input[0] # shape (nt,)
    input[0] = input[0] + roll(input[0], indim) # shape (nt,), data (in_dim, 0 padding, in_dim)
    
    # weight mapping and padding
    it = (indim + outdim - 1) // outdim # = np.ceil(indim / outdim)
    weight = torch.stack([ torch.roll(weight[i,:], -i) for i in range(outdim) ]) # shape (outdim, indim), data just rotated to right
    weight = F.pad(weight, (0, it*outdim - indim) , mode = "constant", value = 0 ) # shape (outdim, it*outdim)
    weight = einops.rearrange(weight, "i1 (i2 i3) -> i3 (i2 i1)", i1 = outdim, i2 = it, i3 = outdim) # transpose
    weight = F.pad(weight, (0, nt - weight.shape[1]), mode = 'constant', value = 0) # shape (outdim, nt)
    
    # rotation and point-wise multiplication
    for i in range(outdim) :
        result[0] = result[0] + roll(input[0], -i) * weight[i, :] # right rotation
    
    # self-sum
    for j in range(cint(np.log2(it))) :
        result[0] = result[0] + roll(result[0], -pow(2, j) * outdim)
        
    # bias padding and addition
    bias = F.pad(bias, (0, nt-bias.shape[0]) , mode = "constant", value = 0 )
    result[0] = result[0] + bias
    return result

def diagCP(input, weight, slot_length) :
    out_dim, inter_dim = weight.shape
    print(f"weight.shape: {weight.shape}")
    mask_input = torch.ones(inter_dim) # shape (inter_dim,)
    mask_input = F.pad(mask_input, (0, slot_length - mask_input.shape[0]), mode= 'constant', value=0) # shape (slot_length,)
    input[0] = mask_input * input[0] # shape (slot_length,)
    
    if (out_dim > inter_dim) :
        div_times = cint(out_dim / inter_dim)
        input, packing_size, packing_times = replicate_ciphertext(input, out_dim, div_times)
        weight = F.pad(weight, (0, out_dim - weight.shape[1]), mode='constant', value=0)
        print(f"out_dim > inter_dim, reshape weight.shape: {out_dim} > {inter_dim}, {weight.shape}")
        out_dim, inter_dim = weight.shape
    else :
        input[0] = input[0] + roll(input[0], inter_dim) # shape (slot_length,)
    
    expand = cint(inter_dim / out_dim)
    weight = torch.stack([ torch.roll(weight[i,:], -i) for i in range(out_dim) ]) # diagonal, (out_dim, inter_dim)
    weight = F.pad(weight, (0, expand*out_dim - inter_dim), mode = 'constant', value = 0) # (out_dim, expand*out_dim)
    weight = einops.rearrange(weight, "i1 (i2 i3) -> i3 (i2 i1)", i1 = out_dim, i2 = expand, i3 = out_dim) # transpose
    weight = F.pad(weight, (0, slot_length - weight.shape[1]), mode = 'constant', value = 0) # (out_dim, slot_length)
   
    result = np.full((1) ,Empty(), dtype = object) # shape (1,)
    for i in range(out_dim) :
        if torch.any(weight[i, :] != 0) :
            if i == 0 :
                tmp = input[0]
            else :
                tmp = roll(tmp, -1)
            result[0] = result[0] + tmp * weight[i, :]
    
    # self-sum
    for j in range(cint(np.log2(expand))) :
        result[0] = result[0] + roll(result[0], -pow(2, j) * out_dim)

    return result