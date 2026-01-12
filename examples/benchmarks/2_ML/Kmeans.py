import hecate as hc
import sys

import poly
from poly.MPCB import *
from poly.Func import *
from poly.Poly import *
import hecate.parser as UTIL

argv = UTIL.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, epochs, input_data = argv

if len(sys.argv) != 1:
    a_epochs = int(epochs)


def sum_elements(data):
    for i in range(1, 9):
        rot = data.rotate(2**i)
        data = data + rot
    return data


def create_mask(elements):
    mask = [[] for _ in range(3)]

    temp = [0.0 for _ in range(elements - 2)]
    temp0 = [0.0 for _ in range(2 * elements - 2)]
    temp1 = [1.0, 1.0]
    temp2 = [1.0, 0.0]

    mask[0] = temp1 + temp0
    mask[1] = temp1 + temp
    mask[2] = temp2

    mask[0] = hc.Plain(mask[0])  # 11000000 ...512
    mask[1] = hc.Plain(mask[1])  # 11000000 ...1024
    mask[2] = hc.Plain(mask[2])  # 10101010 ...
    return mask


def copy(data, mask):
    data = data * mask

    for i in range(1, 9):
        rot = data.rotate(1024 - 2**i)
        data = data + rot

    return data


def newton_inverse(n, epoch):
    # intial guess should be lower than 2/n
    x = hc.Plain([0.01])  # if set low, less chance of divergence
    for _ in range(epoch):
        # with hc.loop(0, epoch, step, inputarr = x) as i:
        x = x * (hc.Plain([2.0]) - n * x)

    return x


@hc.func("c,c")
def Kmeans(x_data, y_data):

    epochs = a_epochs  # this dataset saturates after 5+ epochs
    epochs_newton = 5  # can be set bigger for more accuracy
    elements = 512
    mask = create_mask(elements)

    centroid_0 = copy(x_data, mask[0])
    centroid_1 = copy(x_data.rotate(2), mask[0])

    for _ in range(epochs):
        # calculate the distance difference from centroid to data point
        distance_0 = x_data - centroid_0
        distance_0 = distance_0 * distance_0
        distance_1 = x_data - centroid_1
        distance_1 = distance_1 * distance_1

        # accumulated the distance difference of data points
        dist_diff = distance_0 - distance_1
        dist_diff = dist_diff + dist_diff.rotate(1)

        # shrink dist_diff to fit in HE_sign range
        dist_diff = hc.Plain([0.01]) * dist_diff

        # assign labels depending on the difference
        # sign_diff = hc.Plain([0.5]) + HE_sign(dist_diff)          # has some noise
        sign_diff = hc.Plain([0.5]) + HE_sign(HE_sign(dist_diff))  # more accurate

        label = sign_diff * mask[2]  # only the even indexes are valid

        # calculate the label for each data point
        label_1 = label + label.rotate(elements - 1)
        label_0 = hc.Plain([1.0]) - label_1

        # get data points belonging to each label
        centroid_0 = x_data * label_0
        centroid_1 = x_data * label_1

        # gather number of elements to calculate mean
        sum_0 = sum_elements(label_0)
        sum_1 = sum_elements(label_1)

        # inverse the number of elements
        isum_0 = newton_inverse(sum_0, epochs_newton)
        isum_1 = newton_inverse(sum_1, epochs_newton)

        # calculate mean by sum(data_points) * (1 / num_points)
        centroid_0 = sum_elements(centroid_0) * isum_0
        centroid_1 = sum_elements(centroid_1) * isum_1

    return centroid_0, centroid_1


# @hc.func("c,c,i")
def Kmeans(x_data, y_data):

    epochs = a_epochs  # this dataset saturates after 5+ epochs
    epochs_newton = 5  # can be set bigger for more accuracy
    elements = 512
    mask = create_mask(elements)

    centroid_0 = copy(x_data, mask[0])
    centroid_1 = copy(x_data.rotate(2), mask[0])
    centroid_ = centroid_0 + centroid_1.rotate(elements)

    for _ in range(epochs):
        # calculate the distance difference from centroid to data point
        distance_ = x_data - centroid_
        distance_ = distance_ * distance_

        # accumulated the distance difference of data points
        dist_diff = distance_ - distance_.rotate(elements)
        dist_diff = dist_diff + dist_diff.rotate(1)

        # shrink dist_diff to fit in HE_sign range
        dist_diff = hc.Plain([0.01]) * dist_diff
        # dist_diff = hc.bootstrap(dist_diff)

        # assign labels depending on the difference
        # sign_diff = hc.Plain([0.5]) + HE_sign(dist_diff)          # has some noise
        sign_diff = hc.Plain([0.5]) + HE_sign(HE_sign2(dist_diff))  # more accurate

        sign_diff = hc.bootstrap(sign_diff)
        label = sign_diff * mask[2]  # only the even indexes are valid
        label_ = label + label.rotate(2 * elements - 1)  # copy the even indexes
        label_ = hc.Plain([1.0]) - label_

        # get data points belonging to each label
        centroid_ = x_data * label_

        # gather number of elements to calculate mean
        sum_ = sum_elements(label_)
        sum_ = copy(sum_, mask[1])

        # inverse the number of elements
        isum_ = newton_inverse(sum_, epochs_newton)

        # calculate mean by sum(data_points) * (1 / num_points)
        centroid_ = sum_elements(centroid_) * isum_
        centroid_ = copy(centroid_, mask[1])

    centroid_0 = centroid_
    centroid_1 = centroid_.rotate(elements)
    return centroid_0, centroid_1


modName = hc.save("traced", "traced")
print(modName)
