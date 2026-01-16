import hecate as hc
import sys
import hecate.parser as UTIL

argv = UTIL.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv

if len(sys.argv) != 1:
    a_epochs = int(loop_count)


def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1 << i)
        data += rot

    return data


def poly_y_predict(x0, x1, x2, weight):
    y_predict = x0 * weight[0]
    y_predict += x1 * weight[1]
    y_predict += x2 * weight[2]
    return y_predict


def create_mask_3(elements):
    mask = [[] for i in range(3)]

    temp0 = [0.0 for i in range(elements)]
    temp1 = [1.0 for i in range(elements)]
    temp2 = [0.0 for i in range(elements * 7)]

    mask[0] = temp1 + temp0 + temp0
    mask[1] = temp0 + temp1 + temp0
    mask[2] = temp0 + temp0 + temp1

    mask[0] = mask[0] + mask[0] + mask[0] + temp2
    mask[1] = mask[1] + mask[1] + mask[1] + temp2
    mask[2] = mask[2] + mask[2] + mask[2] + temp2

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])

    return mask


def create_mask_9(elements):
    mask = [[] for i in range(3)]

    temp0 = [0.0 for i in range(elements * 3)]
    temp1 = [1.0 for i in range(elements * 3)]
    temp2 = [0.0 for i in range(elements * 7)]

    mask[0] = temp1 + temp0 + temp0 + temp2
    mask[1] = temp0 + temp1 + temp0 + temp2
    mask[2] = temp0 + temp0 + temp1 + temp2

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])

    return mask


@hc.func("c,c,c,c,c,c,i")
def Multivariate_Loop(x0_data, x1_data, x2_data, y0_data, y1_data, y2_data, epochs):
    learning_rate = hc.Plain([-0.01])
    # epochs = a_epochs

    W0 = [hc.Plain([1.0]) for i in range(3)]
    W1 = [hc.Plain([1.5]) for i in range(3)]
    W2 = [hc.Plain([2.0]) for i in range(3)]
    W = W0 + W1 + W2
    elements = 4096

    # for _ in range(epochs):
    with hc.loop(0, epochs, 1, W, num_elements=elements) as k:
        # wX = [ X[i] * W[j][i] for i in range(3)]

        y_predict0 = W[0] * x0_data + W[1] * x1_data + W[2] * x2_data
        y_predict1 = W[3] * x0_data + W[4] * x1_data + W[5] * x2_data
        y_predict2 = W[6] * x0_data + W[7] * x1_data + W[8] * x2_data

        error0 = y_predict0 - y0_data
        error1 = y_predict1 - y1_data
        error2 = y_predict2 - y2_data

        grad0 = error0 * x0_data
        grad1 = error0 * x1_data
        grad2 = error0 * x2_data
        grad3 = error1 * x0_data
        grad4 = error1 * x1_data
        grad5 = error1 * x2_data
        grad6 = error2 * x0_data
        grad7 = error2 * x1_data
        grad8 = error2 * x2_data
        gradW = [grad0, grad1, grad2, grad3, grad4, grad5, grad6, grad7, grad8]

        gradW = [
            sum_elements(gradW[i]) * hc.Plain([1 / 2048]) * learning_rate
            for i in range(9)
        ]
        W = [W[i] + gradW[i] for i in range(9)]
        W = [hc.bootstrap(W[i]) for i in range(9)]

    return W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8]


# @hc.func("c,c,c,c,c,c,i")
def Multivariate_Loop(x0_data, x1_data, x2_data, y0_data, y1_data, y2_data, epochs):
    step = 1
    learning_rate = hc.Plain([-0.01])

    elements = 4096
    W0 = [1.0 for _ in range(elements * 3)]
    W1 = [1.5 for _ in range(elements * 3)]
    W2 = [2.0 for _ in range(elements * 3)]
    W3 = [0.0 for _ in range(elements * 7)]
    W = W0 + W1 + W2 + W3
    W = hc.Plain(W)  # weights

    mask_3 = create_mask_3(elements)
    mask_9 = create_mask_9(elements)  # mask

    x0_data = x0_data * mask_3[0]
    x1_data = x1_data * mask_3[1]
    x2_data = x2_data * mask_3[2]
    X = x0_data + x1_data + x2_data  # x_data

    y0_data = y0_data * mask_9[0]
    y1_data = y1_data * mask_9[1]
    y2_data = y2_data * mask_9[2]
    Y = y0_data + y1_data + y2_data  # y_data

    # for k in range(epochs):
    with hc.loop(0, epochs, step, W, num_elements=elements) as k:
        wX = W * X
        y_predict = wX + wX.rotate(elements * 1) + wX.rotate(elements * 2)
        mY = -Y
        error = y_predict + mY
        error = error * mask_3[0]
        error0 = X * error
        error1 = X * error.rotate(elements * (16 - 1))
        error2 = X * error.rotate(elements * (16 - 2))
        error = [error0, error1, error2]
        error = [error[i] + error[i].rotate(elements * (16 - 1)) for i in range(3)]
        gradW = [error[i] * hc.Plain([1 / 2048]) for i in range(3)]
        gradW = [sum_elements(gradW[i]) for i in range(3)]

        concat_gradW = gradW[0] * mask_3[0]
        concat_gradW += gradW[1] * mask_3[1]
        concat_gradW += gradW[2] * mask_3[2]
        # concat_gradW = concat_gradW * hc.Plain([1/2048])

        Wup = concat_gradW * learning_rate
        W = W + Wup

        # if a_compile_opt == 16 and (k+1) % 2 == 0 and k != 49:
        #     W = hc.bootstrap(W)

    res = [W.rotate(elements * i) for i in range(9)]

    return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]


modName = hc.save("traced", "traced")
print(modName)
