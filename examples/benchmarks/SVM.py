import hecate as hc
import sys

import poly
from poly.MPCB import *
from poly.Func import *
from poly.Poly import *

def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1<<i)
        data += rot
    
    return data

def create_mask(elements):
    mask = [[] for i in range(4)]

    temp0 = [0.0 for i in range(elements)]
    temp1 = [1.0 for i in range(elements)]
    temp2 = [0.0 for i in range(elements*13)]

    mask[0] = temp1 + temp0 + temp0 + temp2
    mask[1] = temp0 + temp1 + temp0 + temp2
    mask[2] = temp0 + temp0 + temp1 + temp2
    mask[3] = temp1 + temp1 + temp0 + temp2

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])
    mask[3] = hc.Plain(mask[3])

    return mask

# @hc.func("c,c")
# def SVM(x_data, y_data) :
#     epochs = 10
@hc.func("c,c,i")
def SVM(x_data, y_data, epochs) :
    
    step = 1
    learning_rate = hc.Plain([-0.0001])
    lambda_param = hc.Plain([0.02])
    lr_lambda = hc.Plain([-0.00002])

    elements = 4096
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
