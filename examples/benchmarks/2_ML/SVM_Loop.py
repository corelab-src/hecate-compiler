import hecate as hc
import sys

import poly
from poly.MPCB import *
from poly.Func import *
from poly.Poly import *

argv = hc.hc_parser(__file__)
compile_type, waterline, benchmark, library, hardware, epochs, input_data = argv

if(len(sys.argv) != 1):
    a_epochs = int(epochs)

def sum_elements(array, n_samples):
    for i in range(10):
        rot = array.rotate(1<<i)
        array = array + rot
        # array += rot
    return array

def create_mask(elements):
    mask = [[] for i in range(1)]

    temp0 = [0.0 for i in range(elements)]

    mask[0] = temp0
    mask[0] = hc.Plain(mask[0])

    return mask

@hc.func("c,c,c,i")
def SVM_Loop(X0, X1, Y, epochs):

    step = 1
    lambda_ = hc.Plain([0.02])
    learning_rate = hc.Plain([-0.001])
    # lr_lambda = hc.Plain([-0.00002])

    n_samples = 1024
    n_features = 2
    zero = 0.00000001

    W0 = hc.Plain([1.0 for _ in range(n_samples)])
    W1 = hc.Plain([1.0 for _ in range(n_samples)])
    b0 = hc.Plain([1.0 for _ in range(n_samples)])

    W0 = W0 + X0 * hc.Plain([zero])
    W1 = W1 + X0 * hc.Plain([zero])
    b0 = b0 + X0 * hc.Plain([zero])

    minus_one = hc.Plain([-1.0 for _ in range(n_samples)])

    mask = create_mask(n_samples)

    with hc.loop(0, epochs, step, inputarr = [W0, W1, b0], num_elements = n_samples) as i:
        # with hc.loop(0, epochs, step, inputarr = W) as i:
        # calculate the function value
        func = X0 * W0 + X1 * W1 - b0
        func = Y * func + minus_one

        # label the result based on function
        # func = hc.bootstrap(func)
        func = func * hc.Plain([0.1])
        y_predict = HE_sign(HE_sign(func))

        # y_predict = hc.bootstrap(y_predict)
        cond0 =  y_predict + hc.Plain([0.5])
        cond1 = -y_predict + hc.Plain([0.5])

        # correct classification weights are updated to only include the regularization term
        gradW0 = cond0 * lambda_ * W0
        gradW1 = cond0 * lambda_ * W1
        # wrong classification updates the weights
        gradW0 += cond1 * (lambda_ * W0 - X0 * Y)
        gradW1 += cond1 * (lambda_ * W1 - X1 * Y)
        gradb0  = cond1 * Y

        # sum the gradients
        gradW0 = sum_elements(gradW0, n_samples)
        gradW1 = sum_elements(gradW1, n_samples)
        gradb0 = sum_elements(gradb0, n_samples)

        # apply the gradients
        W0 = W0 + gradW0 * learning_rate
        W1 = W1 + gradW1 * learning_rate
        b0 = b0 + gradb0 * learning_rate

        W0 = hc.bootstrap(W0)
        W1 = hc.bootstrap(W1)
        b0 = hc.bootstrap(b0)

    # return gradW0, gradW1, gradb0
    return W0, W1, b0

# @hc.func("c,c,i")
def SVM_Loop(x_data, y_data, epochs) :
    
    step = 1
    learning_rate = hc.Plain([-0.0001])
    lambda_param = hc.Plain([0.02])
    lr_lambda = hc.Plain([-0.00002])

    elements = 1024
    mask = create_mask(elements)
    W = hc.Plain([0.00000001 for _ in range(elements*3)])
    W = W + x_data * hc.Plain([0.00000001])

    # for i in range(epochs):
    with hc.loop(0, epochs, step, inputarr = W) as i:
        dot = W * x_data
        dot = dot + dot.rotate(elements)    # np.dot(x_i, W)
        dot = dot - W.rotate(elements*2)    # np.dot(x_i, W) - b 
        dot = dot * y_data                  # slots 0~4096
        dot = dot + hc.Plain([-1.0])
        dot = dot * hc.Plain([0.01])        # normalize input for sign function
        
        cond = HE_sign(dot)
        cond = hc.bootstrap(cond)
        cond = HE_sign2(cond)
        cond = cond * mask[0]
        cond = cond + cond.rotate(elements*(16-1)) + cond.rotate(elements*(16-2))
        
        cond1 = cond + hc.Plain([0.5])
        cond2 = hc.Plain([-1.0]) * cond + hc.Plain([0.5])
        
        # Wup1 = condition1 * lambda_param * W
        Wup1 = cond1 * lambda_param * W
        Wup1 = Wup1 * mask[3]
        
        # Wup2 = condition2 * (lambda_param * W - x_i * y[idx])
        Wup2 = -x_data
        Wup2 = Wup2 * y_data
        Wup2 = lambda_param * W + Wup2
        Wup2 = cond2 * Wup2
        Wup2 = Wup2 * mask[3]
        bup2 = cond2 * y_data * mask[2]
        
        Wup = (Wup1 + Wup2) * learning_rate
        Wup_1 = Wup * mask[0]
        Wup_2 = Wup * mask[1]
        Wup_1 = Wup_1 + Wup_1.rotate(elements*(16-1))
        Wup_2 = Wup_2 + Wup_2.rotate(elements*(16-1))
        Wup_1 = sum_elements(Wup_1)
        Wup_2 = sum_elements(Wup_2)

        bup = bup2 * learning_rate
        bup = bup + bup.rotate(elements*(16-1))
        bup = sum_elements(bup)

        concat_gradW  = Wup_1 * mask[0]
        concat_gradW += Wup_2 * mask[1]
        concat_gradW += bup   * mask[2] 

        W = W + concat_gradW 

    res = [W.rotate(elements*i) for i in range(3)]
    
    return res[0], res[1], res[2]

modName = hc.save("traced", "traced")
print(modName)
