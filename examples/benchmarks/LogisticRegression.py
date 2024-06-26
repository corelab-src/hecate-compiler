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

# @hc.func("c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c,c")
def LogisticRegression(x_data_0 , x_data_1 , x_data_2 , x_data_3 , x_data_4 , x_data_5 , x_data_6 , x_data_7,
                   x_data_8 , x_data_9 , x_data_10, x_data_11, x_data_12, x_data_13, x_data_14, x_data_15,
                   x_data_16, x_data_17, x_data_18, x_data_19, x_data_20, x_data_21, x_data_22, x_data_23,
                   x_data_24, x_data_25, x_data_26, x_data_27, x_data_28, x_data_29, y_data):

    X = [x_data_0 , x_data_1 , x_data_2 , x_data_3 , x_data_4 , x_data_5 , x_data_6 , x_data_7,
         x_data_8 , x_data_9 , x_data_10, x_data_11, x_data_12, x_data_13, x_data_14, x_data_15,
         x_data_16, x_data_17, x_data_18, x_data_19, x_data_20, x_data_21, x_data_22, x_data_23,
         x_data_24, x_data_25, x_data_26, x_data_27, x_data_28, x_data_29]
    Y = y_data

    n_samples  = 569
    n_features = 30
    block      = 1024
    zero       = 0.0
    ones       = 1.0
    epochs = a_epochs

    learning_rate = hc.Plain([-0.0001])
    learning_rate_avg = hc.Plain([-0.0001/n_samples])

    W = [hc.Plain([0.0000001 for _ in range(block)]) for _ in range(n_features)]
    b = [hc.Plain([0.0000001 for _ in range(block)]) for _ in range(1)]

    mask = create_mask(n_samples, block)
    for i in range(n_features):
        X[i] = X[i] * mask[0]
    Y = Y * mask[0]

    for _ in range(epochs):
        linear_model = hc.Plain([zero])
        for i in range(n_features):
            linear_model = linear_model + X[i] * W[i]
        linear_model = linear_model + b[0]

        # sigmod function
        # linear_model = linear_model * hc.Plain([1/8])
        y_predict = poly.GenPoly()(linear_model) + hc.Plain([0.5])
        # y_predict = hc.bootstrap(y_predict)

        error = [mask[1] for _ in range(n_features)]
        gradW = [mask[1] for _ in range(n_features)]
        gradb = [mask[1] for _ in range(1)]

        # calculate weight gradient
        for i in range(n_features):
            error[i] = (y_predict - Y) * X[i]
            gradW[i] = sum_elements(error[i])
            gradW[i] = gradW[i] * learning_rate_avg

        # calculate bias gradient
        gradb[0] = (y_predict - Y) * mask[0]
        gradb[0] = sum_elements(gradb[0])
        gradb[0] = gradb[0] * learning_rate_avg

        # apply weight and bias gradient
        for i in range(n_features):
            W[i] = W[i] + gradW[i]
        b[0] = b[0] + gradb[0]

   
    return b[0], W[0], W[1], W[2], W[3], W[4], W[5], W[6], W[7], W[8], W[9], W[10], W[11], W[12], W[13], W[14], W[15], W[16], W[17], W[18], W[19], W[20], W[21], W[22], W[23], W[24], W[25], W[26], W[27], W[28], W[29]
 
@hc.func("c,c")
def LogisticRegression(x_data, y_data):
    step = 1

    elements = 569
    block    = 2048 
    learning_rate = hc.Plain([-0.0001])
    learning_rate_elements = hc.Plain([-0.0001/elements])
    W = [0.00000001 for _ in range(block*31)]
    W = hc.Plain(W)
    
    epochs = a_epochs
    mask = create_mask(elements, block)

    for i in range(epochs):
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


# @hc.func("c,c")
# def LogisticRegression(x_data, y_data):
#     epochs = a_epochs
modName = hc.save("traced", "traced")
print (modName)

