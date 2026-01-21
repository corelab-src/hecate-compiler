import hecate as hc
import sys

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, num_test, loop_count, input_data = argv

if len(sys.argv) != 1:
    a_epochs = int(loop_count)


def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1 << i)
        data += rot

    return data


def poly_y_predict(x, weight):
    y_predict = weight[0]
    y_predict += x * weight[1]
    y_predict += x * x * weight[2]
    return y_predict


def poly_y_predict_concat(x_data, W, elements):
    y_predict = W
    y_predict1 = W * x_data
    y_predict += y_predict1.rotate(elements * 1)
    y_predict2 = W * x_data * x_data
    y_predict += y_predict2.rotate(elements * 2)
    return y_predict


def create_mask(elements):
    mask = [[] for i in range(4)]

    temp0 = [0.0 for i in range(elements)]
    temp1 = [1.0 for i in range(elements)]

    mask[0] = temp1 + temp0 + temp0
    mask[1] = temp0 + temp1 + temp0
    mask[2] = temp0 + temp0 + temp1
    mask[3] = temp1 + temp1 + temp0

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])
    mask[3] = hc.Plain(mask[3])

    return mask


@hc.func("c,c,i")
def PolynomialRegression_Loop(x_data, y_data, epochs):
    # epochs = a_epochs
    W = [hc.Plain([1.0]), hc.Plain([1.0]), hc.Plain([1.0])]

    step = 1
    learning_rate = hc.Plain([-0.0001])
    elements = 4096

    # for i in range(epochs):
    with hc.loop(0, epochs, step, inputarr=W, num_elements=elements) as i:

        y_predict = poly_y_predict(x_data, W)
        mY = -y_data
        error0 = y_predict + mY
        error1 = error0 * x_data
        error2 = error0 * x_data * x_data
        error = [error0, error1, error2]
        error = [error[i] * hc.Plain([1 / 2048]) for i in range(3)]
        gradW = [sum_elements(error[i]) for i in range(3)]
        Wup = [gradW[i] * learning_rate for i in range(3)]
        W = [W[i] + Wup[i] for i in range(3)]

    return W[0], W[1], W[2]


# @hc.func("c,c,i")
def PolynomialRegression_Loop(x_data, y_data, epochs):

    W_num = 3
    elements = 4096
    learning_rate = hc.Plain([-0.0001])

    W = hc.Plain([1.0 for i in range(elements * 3)])
    mask = create_mask(elements)  # returns mask in  Plaintext size 4096*3

    # for i in range(epochs):
    with hc.loop(0, epochs, 1, inputarr=W) as i:

        y_predict = poly_y_predict_concat(x_data, W, elements)
        mY = -y_data

        error0 = y_predict + mY
        error0 = error0 * mask[0]
        error0 = error0 + error0.rotate(elements * 2)
        error1 = error0 * x_data
        error2 = error0 * x_data * x_data
        error = [error0, error1, error2]
        gradW = [error[i] * hc.Plain([1 / 2048]) for i in range(3)]
        gradW = [sum_elements(gradW[i]) for i in range(3)]

        concat_gradW = gradW[0] * mask[0]
        concat_gradW1 = gradW[1] * mask[0]
        concat_gradW += concat_gradW1.rotate(elements * 2)
        concat_gradW2 = gradW[2] * mask[0]
        concat_gradW += concat_gradW2.rotate(elements * 1)
        # concat_gradW = concat_gradW * hc.Plain([1/2048])

        Wup = concat_gradW * learning_rate
        W = W + Wup

        # if (i+1) % 2 == 0 and i != 49:
        #     W = hc.bootstrap(W)

    res = [W.rotate(elements * i) for i in range(3)]

    return res[0], res[1], res[2]


modName = hc.save("traced", "traced")
print(modName)
