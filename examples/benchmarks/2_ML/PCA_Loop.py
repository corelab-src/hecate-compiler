import hecate as hc
import sys

import poly
from poly.MPCB import *
from poly.Func import *
from poly.Poly import *


argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv

if len(sys.argv) != 1:
    a_epochs = int(loop_count)
step = 1


def sum_elements1(data, length):
    for i in range(length):
        rot = data.rotate(4 * (1 << i))
        data += rot

    return data


def sum_elements2(data, length):
    for i in range(length):
        rot = data.rotate(1 << i)
        data += rot

    return data


def create_mask(columns, rows):
    mask = [[] for i in range(6)]

    temp0 = [0.0 for i in range(1024 - 4)]
    temp1 = [1.0 for i in range(4)]
    temp2 = [0.0 for i in range(1024 - 600)]
    temp3 = [1.0 for i in range(600)]
    temp4 = [0.0 for i in range(1024 - 1)]
    temp5 = [1.0]
    temp6 = [1.0, 0.0, 0.0, 0.0]
    temp7 = [1.0 for i in range(16)]
    temp8 = [0.0 for i in range(1024 - 16)]

    mask[0] = temp1 + temp0  # 4   ones
    mask[1] = temp3 + temp2  # 600 ones
    mask[2] = temp5 + temp4  # 1   ones
    mask[3] = temp6  #
    mask[4] = temp6 + temp6 + temp6 + temp6 + temp8
    mask[5] = temp7 + temp8

    mask = [hc.Plain(m) for m in mask]

    return mask


def create_bit_mask(length):
    bit_mask = [[] for i in range(4)]

    temp0 = [1.0, 0.0, 0.0, 0.0]
    temp1 = [0.0, 1.0, 0.0, 0.0]
    temp2 = [0.0, 0.0, 1.0, 0.0]
    temp3 = [0.0, 0.0, 0.0, 1.0]
    temp4 = [0.0 for _ in range(length - 4)]

    bit_mask[0] = temp0 + temp4
    bit_mask[1] = temp1 + temp4
    bit_mask[2] = temp2 + temp4
    bit_mask[3] = temp3 + temp4

    bit_mask = [hc.Plain(m) for m in bit_mask]

    return bit_mask


def covariance(x_data, columns, rows, bit_mask):
    cov = hc.Plain([0.0])

    for i in range(1):
        temp = x_data * x_data
        temp = sum_elements1(temp, 8)
        for k in range(4 - i):
            num = temp * bit_mask[k]
            cov += num.rotate(16 - 4 * k - i)

    for i in range(1, 4):
        temp = x_data * x_data.rotate(i)
        temp = sum_elements1(temp, 8)
        for k in range(4 - i):
            num = temp * bit_mask[k]
            cov += num.rotate(16 - 4 * k - i)
            cov += num.rotate(16 - 4 * k - 4 * i)

    cov = cov.rotate(1024 - 16)
    cov = cov * hc.Plain([1 / (rows - 1.0)])
    cov = hc.bootstrap(cov)

    return cov


def copy(x, mask, n):
    x = x * mask
    x = x + x.rotate(1 * n) + x.rotate(2 * n) + x.rotate(3 * n)
    x = x.rotate(1024 - 3 * n)

    return x


def cov_dot(cov, x, mask, n):
    x = copy(x, mask[0], 4)
    x = cov * x
    x = sum_elements2(x, 2)
    x = x * mask[3]
    x = x + x.rotate(3) + x.rotate(6) + x.rotate(9)

    return x


def power_iteration(cov, epoch, sqrt_epoch, columns, mask):

    b_k = mask[0]
    cov_sum = sum_elements1(cov, 2)

    b_k1 = b_k * cov_sum
    b_k1_squared = b_k1 * b_k1
    b_k1_squared = sum_elements2(b_k1_squared, 2)

    b_k1_norm = sqrt_babylonian(b_k1_squared, sqrt_epoch, epochs_newton1)
    b_k1_norm = hc.bootstrap(b_k1_norm)
    b_k1_norm_i = newton_raphson_inverse(b_k1_norm, epochs_newton2)
    b_k1_norm_i = copy(b_k1_norm_i, mask[2], 1)

    b_k = b_k1 * b_k1_norm_i

    # for _ in range(epoch):
    with hc.loop(0, epoch, 1, inputarr=[b_k]) as i:
        b_k1_in = cov_dot(cov, b_k, mask, 4)
        b_k1_squared_in = b_k1_in * b_k1_in
        b_k1_squared_in = sum_elements2(b_k1_squared_in, 2)

        b_k1_norm_in = sqrt_babylonian(b_k1_squared_in, sqrt_epoch, epochs_newton1)
        b_k1_norm_in = hc.bootstrap(b_k1_norm_in)
        b_k1_norm_i_in = newton_raphson_inverse(b_k1_norm_in, epochs_newton2)
        b_k1_norm_i_in = copy(b_k1_norm_i_in, mask[2], 1)

        b_k = b_k1_in * b_k1_norm_i_in

    return b_k


def sqrt_babylonian(n, epoch1, epoch2):
    x = n * hc.Plain([0.5])

    # for _ in range(epoch1):
    with hc.loop(0, epoch1, step, inputarr=[x]) as i:
        x_i = newton_raphson_inverse(x, epoch2)
        a = n * x_i
        b = x + a
        c = b * hc.Plain([0.5])
        x = c

    return x


def newton_raphson_inverse(n, epoch):
    x = hc.Plain([0.1])
    y = hc.Plain([2.0])
    for _ in range(epoch):
        b = n * x * hc.Plain([-1.0])
        x = x * (y + b)

    return x


# epochs_power  = epochs
# epochs_babyl  = 4

epochs_newton1 = 4
epochs_newton2 = 4
epochs_newton3 = 4


@hc.func("c,c,i,i")
def PCA_Loop(x_data, y_data, epoch1, epoch2):
    columns = 4
    rows = 150

    epochs_power = epoch1
    epochs_babyl = epoch2
    mask = create_mask(columns, rows)
    bit_mask = create_bit_mask(1024)

    x_data = x_data * mask[1]
    cov = covariance(x_data, columns, rows, bit_mask)

    eigenvectors = [[] for i in range(4)]
    # eigenvectors = [0.0 for _ in range(16)]
    for i in range(1):

        eigenvector = power_iteration(cov, epochs_power, epochs_babyl, columns, mask)
        # eigenvectors[i] = eigenvector
        # eigenvectors += eigenvector.rotate(1024-4*i)

        # eigenvalue = np.dot(eigenvector.T, np.dot(cov, eigenvector)) / np.dot(eigenvector.T, eigenvector)
        eigenvector = hc.bootstrap(eigenvector)
        dot1 = cov_dot(cov, eigenvector, mask, 4)  # 1 x 4 matrix
        dot2 = eigenvector * dot1
        dot2 = sum_elements2(dot2, 2)
        inv1 = eigenvector * eigenvector
        inv1 = sum_elements2(inv1, 2)  # 1 x 1 matrix
        inv2 = newton_raphson_inverse(inv1, epochs_newton3)

        eigenvalue = dot2 * inv2
        # eigenvalue = hc.bootstrap(eigenvalue)

        copy_eigenvalue = copy(eigenvalue, mask[2], 1)  # 1111xxxxxxxxxxxx
        copy_eigenvalue = copy(copy_eigenvalue, mask[0], 4)  # 1111111111111111
        copy_eigenvector = copy(eigenvector, mask[0], 4)  # 1234123412341234

        for i in range(1, 4):
            element = eigenvector * bit_mask[i]
            eigenvector += element.rotate(1024 - 3 * i)

        long_eigenvector = copy(eigenvector, mask[3], 1)  # 1111222233334444

        del_cov = copy_eigenvalue * copy_eigenvector * long_eigenvector
        cov = cov - del_cov
        cov = cov * mask[5]

    # res = [eigenvector.rotate(columns*i) for i in range(4)]

    # return res
    return eigenvector


modName = hc.save("traced", "traced")
print(modName)
