#!/usr/bin/env python
"""
@file stat.py
@brief Statistical functions for Hecate benchmarks.

This file provides common statistical functions used across various Hecate benchmarks,
including sum, mean, variance, standard deviation, covariance, and correlation
calculations on encrypted data, as well as polynomial approximation methods.
"""
import numpy as np
import hecate as hc


global input_length, num_test, loop_count, padded_length
input_length = hc.input_length
num_test = hc.num_test
loop_count = hc.loop_count
padded_length = hc.padded_length

# ------------------------------------------------------------------
# Basic Statistical Functions
# ------------------------------------------------------------------


def polynomial_evaluation(data, coefficients):
    """
    @brief Evaluate a polynomial at a given point using Horner's Method.

    @param data: encrypted x (hc.Ciphertext object)
    @param coefficients: list of coefficients [a_0, a_1, ..., a_n]

    Use Horner's Method to calculate p(x):
        p(x) = a_0 + x(a_1 + x(a_2 + ... x(a_{n-1} + x(a_n))...))
    """
    # initialize result
    result = coefficients[0]

    for coeff in coefficients[1:]:
        result = result * data
        result = result + coeff

    return result


# sum of the ciphertext
def sum_LRelements(data):
    """
    @brief Sum all elements in a ciphertext using a fixed rotation pattern.

    @param data: encrypted data (hc.Ciphertext object)
    @return: ciphertext with sum in first slot
    """
    for i in range(12):
        rot = data.rotate(1 << (11 - i))
        data = data + rot
    return data


def naive_sum_elements(data, length=hc.padded_length):
    """
    @brief Sum all elements in a ciphertext using a naive approach.

    @param data: encrypted data (hc.Ciphertext object)
    @param length: length of the data
    @return: ciphertext with sum in first slot
    """
    sum = data
    for i in range(length - 1):
        rot = data.rotate(length - i)
        sum = sum + rot
    return sum


def fold_sum_elements(data, length=hc.padded_length):
    """
    @brief Sum all elements in a ciphertext using a folding approach.

    @param data: encrypted data (hc.Ciphertext object)
    @param length: length of the data
    @return: ciphertext with sum in first slot
    """
    iter_count = hc.log_2(length)
    for i in range(iter_count):
        rot = data.rotate(1 << (iter_count - 1 - i))
        data = data + rot
    return data


# def np_fold_sum_elements(data):
#     iter_count = int(np.floor(np.log2(input_length)))
#     for i in range(iter_count):
#         rot = data.rotate(1 << (iter_count - 1 - i))
#         data = data + rot
#     return data


def pad_fold_sum_elements(
    data, padded_length=hc.padded_length, input_length=hc.input_length
):
    """
    @brief Sum all elements in a ciphertext with padding and masking.

    @param data: encrypted data (hc.Ciphertext object)
    @param padded_length: length after padding
    @param input_length: original input length
    @return: ciphertext with sum in first slot
    """
    # ** if padding is true, masking is needed for all operations **
    mask = [1.0] * input_length + [0.0] * (padded_length - input_length)
    data = data * mask
    iter_count = hc.log_2(padded_length)
    for i in range(iter_count):
        rot = data.rotate(1 << (iter_count - 1 - i))
        data = data + rot
    return data


def dict_fold_sum_elements(data, input_length=hc.input_length):
    """
    @brief Sum all elements in a ciphertext using a dictionary-based approach.

    @param data: encrypted data (hc.Ciphertext object)
    @param input_length: length of the data
    @return: ciphertext with sum in first slot
    """
    power_dict = {
        1: 0,
        2: 1,
        4: 2,
        8: 3,
        16: 4,
        32: 5,
        64: 6,
        128: 7,
        256: 8,
        512: 9,
        1024: 10,
        2048: 11,
        4096: 12,
        8192: 13,
        16384: 14,
        32768: 15,
        65536: 16,
        131072: 17,
        262144: 18,
    }
    if input_length in power_dict:
        iter_count = power_dict[input_length]
    else:
        raise ValueError(f"Input length {input_length} not in power_dict")
    for i in range(iter_count):
        rot = data.rotate(1 << (iter_count - 1 - i))
        data = data + rot
    return data


# mean of the ciphertext
def mean_elements(data, input_length, padded_length):
    sum = fold_sum_elements(data, padded_length)
    mean = sum * (1.0 / input_length)
    return mean


# variance of the ciphertext
def var_elements(masked_mean, data, input_length, padded_length):
    """
    @brief Calculate the variance of all elements in a ciphertext.

    @param masked_mean: encrypted mean (hc.Ciphertext object)
    @param data: encrypted data (hc.Ciphertext object)
    @param input_length: length of the data
    @return: ciphertext with variance in first slot
    """
    res_sub = data - masked_mean
    square_sum = fold_sum_elements(res_sub * res_sub, padded_length)
    square_sum = square_sum * (1.0 / input_length)
    return square_sum


# standard deviation of the ciphertext
def std_elements(small_variance, sqrt_factor):
    """
    @brief Calculate the standard deviation of all elements in a ciphertext.

    @param small_variance: encrypted variance (hc.Ciphertext object)
    @param sqrt_factor: scaling factor for the square root
    @param input_length: length of the data
    @return: ciphertext with standard deviation in first slot
    """
    std = sqrt_babylonian(small_variance)
    return std * sqrt_factor


# experimental
def opt_std_elements(small_variance):
    """
    @brief Calculate the standard deviation using an optimized approach.

    @param small_variance: encrypted variance (hc.Ciphertext object)
    @param input_length: length of the data
    @return: ciphertext with standard deviation in first slot
    """
    std = sqrt_inverse_newton_raphson(small_variance)
    std = small_variance * std
    return std


# covariance of the ciphertext
def cov_elements(
    x_data, y_data, masked_x_mean, masked_y_mean, input_length, padded_length
):
    """
    @brief Calculate the covariance between two ciphertexts.

    @param x_data: first encrypted data (hc.Ciphertext object)
    @param y_data: second encrypted data (hc.Ciphertext object)
    @param masked_x_mean: encrypted mean of x_data
    @param masked_y_mean: encrypted mean of y_data
    @param input_length: length of the data
    @return: ciphertext with covariance in first slot
    """
    x_res_sub = x_data - masked_x_mean
    y_res_sub = y_data - masked_y_mean
    product_sum = fold_sum_elements(x_res_sub * y_res_sub, padded_length)
    return product_sum * (1.0 / input_length)


# # experimental
# def opt_cov_elements(x_mean, y_mean, x_data, y_data):
#     x_mul_y = x_data * y_data
#     product_sum = fold_sum_elements(x_mul_y)
#     product_sum = product_sum * (1.0 / input_length)
#     return product_sum - x_mean * y_mean


# correlation of the ciphertext
def corrcoef_elements(covariance, small_x_var, small_y_var, fit_factor, input_length):
    """
    @brief Calculate the correlation coefficient between two ciphertexts.

    @param covariance: encrypted covariance (hc.Ciphertext object)
    @param small_x_var: encrypted variance of x_data
    @param small_y_var: encrypted variance of y_data
    @param fit_factor: scaling factor for the correlation
    @param input_length: length of the data
    @return: ciphertext with correlation coefficient in first slot
    """
    inv_std_x = sqrt_inverse_newton_raphson(small_x_var)
    inv_std_y = sqrt_inverse_newton_raphson(small_y_var)
    inv_std_product = inv_std_x * inv_std_y
    correlation = covariance * inv_std_product
    return correlation * fit_factor


# experimental
def opt_corrcoef_elements(
    covariance, small_x_std, small_y_std, fit_factor, input_length=hc.input_length
):
    """
    @brief Calculate the correlation coefficient using an optimized approach.

    @param covariance: encrypted covariance (hc.Ciphertext object)
    @param small_x_std: encrypted standard deviation of x_data
    @param small_y_std: encrypted standard deviation of y_data
    @param fit_factor: scaling factor for the correlation
    @param input_length: length of the data
    @return: ciphertext with correlation coefficient in first slot
    """
    inv_x_std = inverse_newton_raphson(small_x_std, input_length)
    inv_y_std = inverse_newton_raphson(small_y_std, input_length)
    inv_std_product = inv_x_std * inv_y_std
    correlation = covariance * inv_std_product
    return correlation


# ------------------------------------------------------------------
# Basic Polynomial Approximation Functions
# ------------------------------------------------------------------
def inverse_newton_raphson(data, epochs=5):
    """
    @brief Calculate the inverse of a value using Newton-Raphson method.

    @param data: encrypted data (hc.Ciphertext object)
    @param epochs: number of iterations
    @return: ciphertext with inverse in first slot
    """
    result = 0.5
    plain_2 = 2.0
    plain_minus_1 = -1.0

    for _ in range(epochs):
        intermediate = data * result * plain_minus_1
        result = result * (plain_2 + intermediate)

    return result


def sqrt_babylonian(data, epoch1=5, epoch2=5):
    """
    @brief Calculate the square root using Babylonian method.

    @param data: encrypted data (hc.Ciphertext object)
    @param epoch1: number of outer iterations
    @param epoch2: number of inner iterations for inverse calculation
    @return: ciphertext with square root in first slot
    """
    sqrt_data = data

    for _ in range(epoch1):
        sqrt_data_inv = inverse_newton_raphson(sqrt_data, epoch2)
        a = data * sqrt_data_inv
        b = sqrt_data + a
        c = b * 0.5
        sqrt_data = c

    return sqrt_data


def sqrt_inverse_newton_raphson(data, epochs=5):
    """
    @brief Calculate the inverse square root using Newton-Raphson method.

    @param data: encrypted data (hc.Ciphertext object)
    @param epochs: number of iterations
    @return: ciphertext with inverse square root in first slot
    """
    result = 0.5
    product_result = 0.25
    plain_1_5 = 1.5
    plain_minus_0_5 = -0.5

    for _ in range(epochs):
        intermediate = data * product_result * plain_minus_0_5
        result = result * (plain_1_5 + intermediate)
        product_result = result * result

    return result
