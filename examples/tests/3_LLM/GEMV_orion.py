#!/usr/bin/env python

import faulthandler

faulthandler.enable()

import torch
import numpy as np
import time
import os

import hecate as hc
import hetorch as ht

# ------------------------------------------------------------------
# GEMV Python Function
# ------------------------------------------------------------------
# slot_length = 2 ** (17-1) # HEaaN GPU
slot_length = 2 ** (16 - 1)  # HEONGPU


def python_original(input, weight, bias):
    print("================================================")
    print("input configuration")
    print("================================================")
    # print("input: ", input)
    print(f"{'dtype':<15} {'device':<7} {'shape'}")
    print(f"{str(input.dtype):<15} {str(input.device):<7}", input.shape)
    print(f"{'max':<10} {'min':<10} {'mean':<10} {'std':<10}")
    print(
        f"{input.max().item():<10.5g} {input.min().item():<10.5g} {input.mean().item():<10.5g} {input.std().item():<10.5g}"
    )

    if input.device.type == "cuda":
        print("Input is on GPU, do matmul on GPU")
    else:
        print("Input is on CPU, do matmul on CPU")
    print("================================================")
    print("Input * Weight + Bias is: ")
    print(input.shape, "*", weight.shape, "+", bias.shape, "=", bias.shape)
    print("================================================")
    return torch.matmul(input, weight)


# ------------------------------------------------------------------
# Main Function
# ------------------------------------------------------------------
global IN_DIM, OUT_DIM, INTER_DIM

if __name__ == "__main__":
    argv = hc.hc_parser(__file__)
    compile_opt, waterline, benchmark, library, hardware, epochs, input_data = argv

    # input, weight, bias = ht.json_to_data(benchmark)
    input, weight, bias = ht.binary_to_data(benchmark)
    IN_DIM, OUT_DIM, INTER_DIM = input.shape[0], weight.shape[0], weight.shape[1]
    print("IN_DIM: ", IN_DIM, "OUT_DIM: ", OUT_DIM, "INTER_DIM: ", INTER_DIM)

    # warm up
    a_input, a_weight, a_bias = input, weight.T, bias.T
    # reference execution time
    ref_timer_start = time.perf_counter_ns()
    reference = python_original(a_input, a_weight, a_bias)
    ref_execution_time = (time.perf_counter_ns() - ref_timer_start) / pow(10, 9)
    reference = reference.to(torch.device("cpu"))

    mem_before = hc.print_mem("Before hevm.run()")
    hc.setLibnHW(argv)  # Set the path to the libnHW.so
    hevm = hc.HEVM()
    hevm.load(
        f"traced/cst/_hecate_{benchmark}.cst",
        f"optimized/{compile_opt}/{benchmark}.{waterline}._hecate_{benchmark}.hevm",
    )
    import torch.nn.functional as F

    input = F.pad(input, (0, 2**15 - input.shape[0]), mode="constant", value=0)
    print("input.shape: ", input.shape)
    [hevm.setInput(i, input[i, :]) for i in range(input.shape[0])]
    timer_start = time.perf_counter_ns()
    hevm.run()
    execution_time = (time.perf_counter_ns() - timer_start) / pow(10, 9)
    mem_after = hc.print_mem("After hevm.run()")
    mem_diff = mem_after - mem_before

    print("================================================")
    print("Evaluation result")
    print("================================================")
    res_he = hevm.getOutput()
    # ref_tensor = torch.from_numpy(reference)
    ref_tensor = reference
    reference_shape = ref_tensor.shape
    # ref_tensor = ref_tensor.reshape(reference_shape)
    res_he = res_he[: reference_shape[0], : reference_shape[1]]
    result_tensor = torch.from_numpy(res_he)

    # Check if the shapes match before final comparison
    assert result_tensor.shape == ref_tensor.shape, (
        "result_tensor.shape: "
        + str(result_tensor.shape)
        + " != ref_tensor.shape: "
        + str(ref_tensor.shape)
    )
    rms = torch.sqrt(torch.mean(torch.pow(result_tensor - ref_tensor, 2)))
    hevm.printer(execution_time, rms.item())
    print("ref_execution_time(s): ", ref_execution_time)
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
