#!/usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import einops
from pathlib import Path
import os
import math

import hecate as hc
from poly.models.ResNet import resnet20
import poly.MPCB as MPCB
import poly.Func as Func
import hetorch as ht

# ------------------------------------------------------------------
# Global Variables
# ------------------------------------------------------------------
model = None
N = 2**15  # Number of slots

source_path = Path(__file__).resolve()
source_dir = source_path.parent
hecate_dir = os.environ["HECATE"]
cache_dir = str(hecate_dir) + "/examples/data/cache_resnet20_silu"

def getModel():
    model = resnet20()
    model_dict = torch.load(str(hecate_dir)+"/examples/data/resnet20_silu_model", map_location=torch.device('cpu'))
    model.load_state_dict(model_dict['state_dict'])
    model = model.eval()
    return model

ht.conv_gemv_cp = ht.bsgsCP_orion_ptwo
fc_gemv_cp = ht.bsgsCP_orion_fc

# ------------------------------------------------------------------
# Main Hecate Function
# ------------------------------------------------------------------
@hc.func("c")
def ResNet_orion(*ctxt):
    global model
    
    def act(input):
        return Func.HE_SiLU(input)
    
    def bootstrap(out):
        out[0] = hc.bootstrap(out[0])
        return out
    
    input_shape = (3, 32, 32)
    scale_factor = 32
    current_gap = 1
    initial_shapes = {
        # Constant
        "nt" : N,
        "bb" : scale_factor,
        # Input Characteristics (Cascaded)
        "ko" : 1,
        "ho" : 32,
        "wo" : 32
    }
    
    mpp_vec = np.empty((1,), dtype=object)
    mpp_vec[0] = ctxt[0]
    conv1 = model.conv1
    out, out_shape = ht.orion_conv2d_cached(mpp_vec, conv1.weight.detach(), conv1.stride[0], conv1.padding[0], input_shape, "conv1", cache_dir, is_caching=True)
    out = bootstrap(out)
    out = ht.orion_bn(out, model.bn1, out_shape, scale_factor)
    out = act (out)
        
    for i in range(0, len(model.layer1)) :
        print(f"\n layer1.{i} block conv1\n")
        residual_connection = out
        layer_conv1 = model.layer1[i].conv1
        out, out_shape = ht.orion_conv2d_cached(out, layer_conv1.weight.detach(), layer_conv1.stride[0], layer_conv1.padding[0], out_shape, f"layer1_{i}_conv1", cache_dir, is_caching=True)
        out = bootstrap(out)
        out = ht.orion_bn(out, model.layer1[i].bn1, out_shape, scale_factor)
        out = act (out)

        print(f"\n layer1.{i} block conv2\n")
        layer_conv2 = model.layer1[i].conv2
        out, out_shape = ht.orion_conv2d_cached(out, layer_conv2.weight.detach(), layer_conv2.stride[0], layer_conv2.padding[0], out_shape, f"layer1_{i}_conv2", cache_dir, is_caching=True)
        out = ht.orion_bn(out, model.layer1[i].bn2, out_shape, scale_factor)
        out = out + residual_connection
        out = bootstrap(out)
        out = act (out)
    
    for i in range(len(model.layer2)):
        shortcut_out = np.full((1,), None, dtype = object)
        shortcut_out[0] = out[0]
        residual_shape, residual_gap = out_shape, current_gap

        print(f"\n layer2.{i} block conv1\n") # First conv (stride=2, gap 1->2)
        layer_conv1 = model.layer2[i].conv1
        out, out_shape, current_gap = ht.orion_conv2d_cached_with_gap(
            out, layer_conv1.weight.detach(), layer_conv1.stride[0], 
            layer_conv1.padding[0], out_shape, 
            cache_name=f"layer2_{i}_conv1", cache_dir=cache_dir, is_caching=True, input_gap=current_gap, output_gap=2)
        out = bootstrap(out)
        out = ht.orion_bn_with_gap(out, model.layer2[i].bn1, out_shape, current_gap)
        out = act (out)
        
        if i == 0: 
            print(f"Applying shortcut: input shape {residual_shape} (gap={residual_gap}) -> output shape {out_shape} (gap={current_gap})")
            shortcut_module = model.layer2[i].shortcut
            shortcut_out, shortcut_shape, shortcut_gap = ht.orion_shortcut_conv2d_with_gap(
                shortcut_out, shortcut_module, residual_shape,
                input_gap=residual_gap, output_gap=current_gap,
                cache_name=f"layer2_{i}_shortcut", cache_dir=cache_dir, is_caching=True
        )

        print(f"\n layer2.{i} block conv2\n") # Second conv (stride=1, gap 2->2)
        layer_conv2 = model.layer2[i].conv2
        out, out_shape, current_gap = ht.orion_conv2d_cached_with_gap(
            out, layer_conv2.weight.detach(), layer_conv2.stride[0],
            layer_conv2.padding[0], out_shape,
            cache_name=f"layer2_{i}_conv2", cache_dir=cache_dir, is_caching=True, input_gap=current_gap, output_gap=current_gap)
        out = ht.orion_bn_with_gap(out, model.layer2[i].bn2, out_shape, current_gap)
        out = out + shortcut_out
        out = bootstrap(out)
        out = act (out)

    for i in range(len(model.layer3)):
        shortcut_out[0] = out[0]
        residual_shape, residual_gap = out_shape, current_gap

        print(f"\n layer3.{i} block conv1\n") # First conv (stride=2, gap 2->4)
        layer_conv1 = model.layer3[i].conv1
        out, out_shape, current_gap = ht.orion_conv2d_cached_with_gap(
            out, layer_conv1.weight.detach(), layer_conv1.stride[0], 
            layer_conv1.padding[0], out_shape, 
            cache_name=f"layer3_{i}_conv1", cache_dir=cache_dir, is_caching=True, input_gap=current_gap, output_gap=4)
        out = bootstrap(out)
        out = ht.orion_bn_with_gap(out, model.layer3[i].bn1, out_shape, current_gap)
        out = act (out)
        
        if i == 0: 
            print(f"Applying shortcut: input shape {residual_shape} (gap={residual_gap}) -> output shape {out_shape} (gap={current_gap})")
            shortcut_module = model.layer3[i].shortcut
            shortcut_out, shortcut_shape, shortcut_gap = ht.orion_shortcut_conv2d_with_gap(
                shortcut_out, shortcut_module, residual_shape,
                input_gap=residual_gap, output_gap=current_gap,
                cache_name=f"layer3_{i}_shortcut", cache_dir=cache_dir, is_caching=True
            )
        
        print(f"\n layer3.{i} block conv2\n") # Second conv (stride=1, gap 4->4)
        layer_conv2 = model.layer3[i].conv2
        out, out_shape, current_gap = ht.orion_conv2d_cached_with_gap(
            out, layer_conv2.weight.detach(), layer_conv2.stride[0],
            layer_conv2.padding[0], out_shape,
            cache_name=f"layer3_{i}_conv2", cache_dir=cache_dir, is_caching=True, input_gap=current_gap, output_gap=current_gap)
        out = ht.orion_bn_with_gap(out, model.layer3[i].bn2, out_shape, current_gap)
        out = out + shortcut_out
        out = bootstrap(out)
        out = act (out)
    
    print("avgpool")
    out, out_shape, current_gap = ht.orion_avgpool_global(out, out_shape, current_gap, 32, "layer4_avgpool", cache_dir, is_caching=True)
    
    print("linear")
    out = fc_gemv_cp(out, model.linear.weight, initial_shapes["nt"])
    out[0] = out[0] + model.linear.bias.cpu() / scale_factor
    
    print("ResNet_orion trace is done")
    
    return out

# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
if __name__ == "__main__":
    # Load model
    model = getModel()
    model = model.type(torch.double)
    model = model.cpu()
    
    # Save traced module
    module_name = hc.save("traced", "traced")