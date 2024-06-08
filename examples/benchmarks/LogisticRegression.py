import hecate as hc
import sys

import poly
from poly.MPCB import *
from poly.Func import *

if(len(sys.argv) != 1):
    a_epochs = int(sys.argv[1])
def sum_elements(data):
    for i in range(10):
        rot = data.rotate(1<<i)
        data += rot

    return data

def create_mask(elements, block):
    mask = [[] for i in range(10)]
    
    temp0 = [1.0 for i in range(elements)]
    temp1 = [0.0 for i in range(block-elements)]
    temp2 = [0.0 for i in range(block)]
    
    mask[0] = temp0 + temp1
    mask[1] = temp0

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    return mask

def LR_y_predict(x_data, W):
    dot = x_data * W
    y_predict = hc.Plain([0.0])
    for i in range(31):
        y_predict = dot.rotate(2048*i)
    return y_predict

# @hc.func("c,c")
# def LogisticRegression(x_data, y_data):
#     epochs = a_epochs
@hc.func("c,c,i")
def LogisticRegression(x_data, y_data, epochs):
    step = 1

    elements = 569
    block    = 2048 
    learning_rate = hc.Plain([-0.0001])
    learning_rate_elements = hc.Plain([-0.0001/elements])
    W = [0.00000001 for _ in range(block*31)]
    W = hc.Plain(W)
    
    mask = create_mask(elements, block)

    # for i in range(epochs):
    with hc.loop(0, epochs, step, inputarr = W) as i:
        y_predict = LR_y_predict(x_data, W)
        y_predict = y_predict * hc.Plain([1/8])
        y_predict = poly.GenPoly()(y_predict) + hc.Plain([0.5])
        y_predict = hc.bootstrap(y_predict)
        
        error = y_predict - y_data
        error = error * x_data
        error = error + error.rotate(1024*(64-1))
        
        grad = error * learning_rate_elements
        grad = sum_elements(grad)
        grad = grad * mask[0] 

        W = W + grad

    res = [W.rotate(block*i) for i in range(31)]
    
    return res

modName = hc.save("traced", "traced")
print (modName)

