#!/usr/bin/env python

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from pathlib import Path
import time
from random import seed
import einops
import math
import os

# Import necessary functions
import hecate as hc
import poly.MPCB as MPCB
from poly.models.ResNet import resnet20
import hetorch as ht

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
    model = resnet20()
    model_dict = torch.load(
        str(hecate_dir) + "/examples/data/resnet20_silu_model",
        map_location=torch.device("cpu"),
    )
    model.load_state_dict(model_dict["state_dict"])
    return model.eval()


def process_reference(x):
    x = x.to("cpu")
    model = getModel().to(x.device)

    with torch.no_grad():
        out = model(x)
        # out = model.module.conv1(x)
        # out = model.module.bn1(out)
        # out = model.module.mish(out)
        # out = model.module.layer1(out)

        print("Reference output shape: ", out.shape)
    return out.flatten().cpu().numpy(), out.shape


if __name__ == "__main__":
    print("ht-based ResNet20 Test")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = getModel()

    # Setup HEVM
    mem_before = hc.print_mem("Before hevm.run()")
    hc.setLibnHW(argv)
    hevm = hc.HEVM()
    stem = Path(__file__).stem
    hevm.load(
        f"traced/cst/_hecate_{stem}.cst",
        f"optimized/{compile_type}/{stem}.{waterline}._hecate_{stem}.hevm",
    )

    # Get test input
    (input_img, target) = val_loader.dataset[0]
    input_var = input_img.unsqueeze(0)
    scale_factor = 32

    # Get reference output
    reference, reference_shape = process_reference(input_var)
    print(f"Reference shape: {reference.shape}")
    print(f"Reference prediction: {np.argmax(reference)}")

    # Pack input for HEVM
    packed_input = ht.orion_preprocess(input_var, scale_factor)
    print(f"Packed input shape: {packed_input.shape}")
    print("\nRunning encrypted inference...")
    execution_time = 0
    for _ in range(epochs):
        hevm.setInput(0, packed_input)
        timer_start = time.perf_counter_ns()
        hevm.run()
        timer_end = (time.perf_counter_ns() - timer_start) / pow(10, 9)
        execution_time += timer_end
    execution_time /= epochs
    mem_after = hc.print_mem("After hevm.run()")
    mem_diff = mem_after - mem_before

    res_he = hevm.getOutput()
    ref_tensor = torch.from_numpy(reference)
    ref_tensor = ref_tensor.reshape(reference_shape)
    result_tensor = torch.from_numpy(res_he)

    # # layer1
    # result_tensor = ht.orion_postprocess(result_tensor, reference_shape)

    # # layer2
    # final_output_gap = 2
    # expected_clear_shape = torch.Size([1, 32, 16, 16])
    # result_tensor = ht.orion_multiplexed_postprocess(
    #     result_tensor.unsqueeze(0) if result_tensor.dim() == 1 else result_tensor,
    #     expected_clear_shape,
    #     gap=final_output_gap,
    #     scale_factor=scale_factor
    # )

    # # layer3
    # final_output_gap = 4
    # expected_clear_shape = torch.Size([1, 64, 8, 8]) # layer3
    # result_tensor = ht.orion_multiplexed_postprocess(
    #     result_tensor.unsqueeze(0) if result_tensor.dim() == 1 else result_tensor,
    #     expected_clear_shape,
    #     gap=final_output_gap,
    #     scale_factor=scale_factor
    # )

    # avgpool, linear (gap = 1)
    final_output_gap = 1
    # expected_clear_shape = torch.Size([1, 64, 1, 1]) # avgpool
    expected_clear_shape = torch.Size([1, 10, 1, 1])  # linear
    result_tensor = ht.orion_multiplexed_postprocess(
        result_tensor.unsqueeze(0) if result_tensor.dim() == 1 else result_tensor,
        expected_clear_shape,
        gap=final_output_gap,
        scale_factor=1,
    )
    ref_tensor = ref_tensor / scale_factor

    # apply all postprocess
    result_tensor = result_tensor.reshape(reference_shape)

    # Check if the shapes match before final comparison
    assert result_tensor.shape == ref_tensor.shape, (
        "result_tensor.shape: "
        + str(result_tensor.shape)
        + " != ref_tensor.shape: "
        + str(ref_tensor.shape)
    )
    rms = torch.sqrt(torch.mean(torch.pow(result_tensor - ref_tensor, 2)))
    hevm.printer(execution_time, rms.item())
    print("CPU memory usage: ", mem_diff, "GB")

    # Calculate error percentage (add small epsilon to avoid division by zero)
    diff = (result_tensor - ref_tensor).abs()
    bits = -torch.log2(diff)
    epsilon = 1e-9
    error_percentage = diff / (ref_tensor.abs() + epsilon) * 100

    # # Find the maximum error value position (multi-dimensional support)
    # max_err_idx_flat = torch.argmax(error_percentage)
    # max_err_idx = np.unravel_index(max_err_idx_flat.cpu().numpy(), error_percentage.shape)

    # print("\n================================================")
    # print("Evaluation Result")
    # print("================================================")
    # print(f"HE raw output shape:      {res_he.shape}")
    # print(f"Reference tensor shape:   {ref_tensor.shape}")
    # print(f"Processed result shape:   {result_tensor.shape}")
    # # np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True, floatmode='fixed', precision=7)
    # # print(f"res_he: {res_he[0,:128]}")
    # # print(f"Reference tensor:         {ref_tensor}")
    # # print(f"Processed result:         {result_tensor}")
    # print(f"ref (max, min, mean, std): {ref_tensor.max():.7g}, {ref_tensor.min():.7g}, {ref_tensor.mean():.7g}, {ref_tensor.std():.7g}")
    # print(f"res (max, min, mean, std): {result_tensor.max():.7g}, {result_tensor.min():.7g}, {result_tensor.mean():.7g}, {result_tensor.std():.7g}")
    bits_rms = torch.sqrt(torch.mean(bits**2))
    error_rms = torch.sqrt(torch.mean(error_percentage**2))
    print(
        f"bits (max, min, rms, std): {bits.max():.7g}, {bits.min():.7g}, {bits_rms:.7g}, {bits.std():.7g}"
    )
    print(
        f"error% (max, min, rms, std): {error_percentage.max():.7g}, {error_percentage.min():.7g}, {error_rms:.7g}, {error_percentage.std():.7g}"
    )

    # print("================================================")
    # print("Error distribution analysis:")
    # print("================================================")
    # print(f"Total elements: {error_percentage.numel()}")
    # print(f"Errors > 0.1%: {(error_percentage > 0.01).sum().item()}")
    # print(f"Errors > 1%: {(error_percentage > 0.1).sum().item()}")
    # print(f"Errors > 5%: {(error_percentage > 1).sum().item()}")
    # print("================================================")
    # # torch.set_printoptions(precision=4, threshold=float('inf'), linewidth=200)
    # # print(error_percentage)
    # # torch.set_printoptions(profile='default')
    # print(f"{'Max Error percentage:':<25} {torch.max(error_percentage).item():.7g}%")
    # print(f"{'Min Error percentage:':<25} {torch.min(error_percentage).item():.7g}%")
    # print(f"{'Mean Error percentage:':<25} {torch.mean(error_percentage).item():.7g}%")
    # # print("\n------------------------------------------------")
    # # print("Maximum Error Details")
    # # print("------------------------------------------------")
    # # print(f"{'Index:':<18} {list(max_err_idx)}")
    # # print(f"{'HE Result:':<18} {result_tensor[max_err_idx].item():.7g}")
    # # print(f"{'Reference:':<18} {ref_tensor[max_err_idx].item():.7g}")
    # # print(f"{'Difference:':<18} {diff[max_err_idx].item():.7g}")
    # # print(f"{'Error Percentage:':<18} {error_percentage[max_err_idx].item():.7g}%")

    # target = torch.tensor([target])
    # criterion = nn.CrossEntropyLoss()
    # loss_ref = criterion(ref_tensor, target)
    # loss_res = criterion(result_tensor, target)
    # print(f"Loss (ref): {loss_ref.item()}")
    # print(f"Loss (res): {loss_res.item()}")
    # print("\n------------------------------------------------")
    # print("Final Prediction Result")
    # print("------------------------------------------------")
    # pred_ref = torch.argmax(ref_tensor)
    # pred_he = torch.argmax(result_tensor)
    # is_correct_ref = "Correct" if pred_ref.item() == target else "Incorrect"
    # is_correct_he = "Correct" if pred_he.item() == target else "Incorrect"
    # print(f"{'True Label (Target):':<25} {target}")
    # print(f"{'Reference Model Prediction:':<25} {pred_ref.item()} ({is_correct_ref})")
    # print(f"{'HE Model Prediction:':<25} {pred_he.item()} ({is_correct_he})")
    # print("================================================\n")

    # torch.set_printoptions(profile="full")
    # # print(f"HE raw output:            {res_he}")
    # saved_path = str(source_dir)+"/../../AlexNet.pt"
    # import os
    # # delete the file if it exists
    # if os.path.exists(saved_path):
    #     os.remove(saved_path)
    # torch.save(res_he, saved_path)
