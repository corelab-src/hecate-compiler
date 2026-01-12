import hecate as hc
import sys
import poly
import os
from poly.models.ResNet_relu import *
from poly.MPCB import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn.functional as F
from PIL import Image
import numpy as np
from random import *
import pprint


from pathlib import Path

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, epochs, input_data = argv

seed(100)
source_path = Path(__file__).resolve()
source_dir = source_path.parent
hecate_dir = os.environ["HECATE"]

# Data loading
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
val_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10(
        root=str(hecate_dir) + "/examples/data/CIFAR10",
        train=False,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    ),
    batch_size=128,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)


def getModel():
    model = torch.nn.DataParallel(resnet20())
    model_dict = torch.load(
        str(hecate_dir) + "/examples/data/resnet20.silu.model",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(model_dict["state_dict"])
    return model.eval()


def preprocess(x):
    import json

    lib_name = library
    hw_name = hardware
    config_name = f"profiled_{lib_name}_{hw_name}.json"
    with open(str(hecate_dir) + "/" + config_name, "r") as f:
        config = json.load(f)
    initial_shapes = {
        # Constant
        # "nt" : 2**16,
        # "nt" : 2**14,
        "nt": config["polynomialDegree"] >> 1,
        "bb": 32,
        # Input Characteristics (Cascaded)
        "ko": 1,
        "ho": 32,
        "wo": 32,
    }
    conv1_shapes = CascadeConv(initial_shapes, model.module.conv1)
    close = shapeClosure(**conv1_shapes)
    return close["MPP"](x)[0]


def process(x):
    x = x.to("cpu")
    model = getModel().to(x.device)

    with torch.no_grad():
        out = model.module(x)

        # out = model.module.conv1(x)
        # out = model.module.bn1(out)
        # out = model.module.mish(out)
        # out = model.module.layer1(out)
        # out = model.module.layer2(out)
        # out = model.module.layer3(out)
        # out = F.avg_pool2d(out, out.size()[3])
        # out = out.view(out.size(0), -1)

        # out = model.module.layer2[0].shortcut(out)
        # for i in range(0, len(model.module.layer2)):
        #     residual_connection = out
        #     out = model.module.layer2[i].conv1(out)
        #     out = model.module.layer2[i].bn1(out)
        #     out = model.module.layer2[i].mish(out)
        #     out = model.module.layer2[i].conv2(out)
        #     out = model.module.layer2[i].bn2(out)
        #     if i == 0:
        #         shortcut_out = out
        #     else:
        #         shortcut_out = residual_connection
        #     out = out + shortcut_out
        #     out = model.module.mish(out)

    return out.flatten().cpu().numpy(), out.shape


def postprocess(flattened_output, output_shape, scale_factor=32):
    torch_res_size = 1
    for i in range(len(output_shape)):
        torch_res_size *= output_shape[i]
    return flattened_output[0, :torch_res_size].reshape(output_shape) * scale_factor


if __name__ == "__main__":

    from random import *
    import sys
    from pathlib import Path
    import time
    from PIL import Image

    model = getModel()
    model = model.eval()

    a_compile_type = compile_type
    a_compile_opt = int(waterline)
    hc.setLibnHW(argv)
    mem_before = hc.print_mem("Before hevm.run()")
    stem = Path(__file__).stem
    hevm = hc.HEVM()
    stem = Path(__file__).stem
    hevm.load(
        f"traced/_hecate_{stem}.cst",
        f"optimized/{a_compile_type}/{stem}.{a_compile_opt}._hecate_{stem}.hevm",
    )

    (input, target) = val_loader.dataset[0]
    input_var = input.unsqueeze(0)
    target = torch.tensor([target])
    target_var = target
    reference, reference_shape = process(input_var)
    print("input_var.shape", input_var.shape)
    print("preprocess(input_var)", preprocess(input_var))
    print("preprocess(input_var).shape", preprocess(input_var).shape)
    resnet_input = [
        hevm.setInput(i, dat) for i, dat in enumerate([preprocess(input_var)])
    ]
    print("resnet_input", resnet_input)
    print("resnet_input.shape", len(resnet_input))
    timer_start = time.perf_counter_ns()
    hevm.run()
    execution_time = (time.perf_counter_ns() - timer_start) / pow(10, 9)
    mem_after = hc.print_mem("After hevm.run()")
    mem_diff = mem_after - mem_before

    res_he = hevm.getOutput()
    ref_tensor = torch.from_numpy(reference)
    ref_tensor = ref_tensor.reshape(reference_shape)
    result_he = torch.from_numpy(res_he)
    # result_tensor = orion_postprocess(result_he, reference_shape, scale_factor)
    # result_tensor = orion_multiplexed_postprocess(result_he, reference_shape, scale_factor)
    result_tensor = postprocess(result_he, reference_shape)

    # Check if the shapes match before final comparison
    assert result_tensor.shape == ref_tensor.shape, (
        "result_tensor.shape: "
        + str(result_tensor.shape)
        + " != ref_tensor.shape: "
        + str(ref_tensor.shape)
    )

    # Calculate error percentage (add small epsilon to avoid division by zero)
    diff = (result_tensor - ref_tensor).abs()
    bits = -torch.log2(diff)
    epsilon = 1e-9
    error_percentage = diff / (ref_tensor.abs() + epsilon) * 100

    # Find the maximum error value position (multi-dimensional support)
    max_err_idx_flat = torch.argmax(error_percentage)
    max_err_idx = np.unravel_index(
        max_err_idx_flat.cpu().numpy(), error_percentage.shape
    )

    print("\n================================================")
    print("Evaluation Result")
    print("================================================")
    print(f"HE raw output shape:      {res_he.shape}")
    print(f"Reference tensor shape:   {ref_tensor.shape}")
    print(f"Processed result shape:   {result_tensor.shape}")
    print(f"Reference tensor:         {ref_tensor}")
    print(f"Processed result:         {result_tensor}")
    print(
        f"ref (max, min, mean, std): {ref_tensor.max():.7g}, {ref_tensor.min():.7g}, {ref_tensor.mean():.7g}, {ref_tensor.std():.7g}"
    )
    print(
        f"res (max, min, mean, std): {result_tensor.max():.7g}, {result_tensor.min():.7g}, {result_tensor.mean():.7g}, {result_tensor.std():.7g}"
    )
    print(
        f"bits (max, min, mean, std): {bits.max():.7g}, {bits.min():.7g}, {bits.mean():.7g}, {bits.std():.7g}"
    )
    print(
        f"{'err% (max, min, mean, std):':<18} {error_percentage.max():.7g}, {error_percentage.min():.7g}, {error_percentage.mean():.7g}, {error_percentage.std():.7g}"
    )

    print("\n------------------------------------------------")
    print("Maximum Error Details")
    print("------------------------------------------------")
    print(f"{'Index:':<18} {list(max_err_idx)}")
    print(f"{'HE Result:':<18} {result_tensor[max_err_idx].item():.7g}")
    print(f"{'Reference:':<18} {ref_tensor[max_err_idx].item():.7g}")
    print(f"{'Difference:':<18} {diff[max_err_idx].item():.7g}")
    print(f"{'Error Percentage:':<18} {error_percentage[max_err_idx].item():.7g}%")

    # Calculate RMS error
    rms = torch.sqrt(torch.mean(torch.pow(result_tensor - ref_tensor, 2)))

    hevm.printer(execution_time, rms.item())

    torch.set_printoptions(profile="full")
    # print(f"HE raw output:            {result_he}")
    saved_path = str(source_dir) + "/../../ResNet_orin.pt"
    import os

    # delete the file if it exists
    if os.path.exists(saved_path):
        os.remove(saved_path)
    torch.save(result_he, saved_path)
