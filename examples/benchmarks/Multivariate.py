import hecate as hc 
import sys

def sum_elements(data):
    for i in range(12):
        rot = data.rotate(1<<i)
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
    temp2 = [0.0 for i in range(elements*7)]
    
    mask[0] = temp1 + temp0 + temp0
    mask[1] = temp0 + temp1 + temp0
    mask[2] = temp0 + temp0 + temp1

    mask[0] = mask[0]+mask[0]+mask[0]+temp2
    mask[1] = mask[1]+mask[1]+mask[1]+temp2
    mask[2] = mask[2]+mask[2]+mask[2]+temp2

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])

    return mask

def create_mask_9(elements):
    mask = [[] for i in range(3)]

    temp0 = [0.0 for i in range(elements*3)]
    temp1 = [1.0 for i in range(elements*3)]
    temp2 = [0.0 for i in range(elements*7)]

    mask[0] = temp1 + temp0 + temp0 + temp2
    mask[1] = temp0 + temp1 + temp0 + temp2
    mask[2] = temp0 + temp0 + temp1 + temp2

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])

    return mask

# @hc.func("c,c,c,c,c,c")
# def Multivariate (x0_data, x1_data, x2_data, y0_data, y1_data, y2_data) :
#     epochs = 10
@hc.func("c,c,c,c,c,c,i")
def Multivariate (x0_data, x1_data, x2_data, y0_data, y1_data, y2_data, epochs) :
    step = 1
    learning_rate = hc.Plain([-0.01])

    elements = 4096
    W0 = [1.0 for _ in range(elements*3)]
    W1 = [1.5 for _ in range(elements*3)]
    W2 = [2.0 for _ in range(elements*3)]
    W3 = [0.0 for _ in range(elements*7)]
    W = W0 + W1 + W2 + W3
    W = hc.Plain(W)                     # weights

    mask_3 = create_mask_3(elements)
    mask_9 = create_mask_9(elements)    # mask
    
    x0_data = x0_data * mask_3[0]
    x1_data = x1_data * mask_3[1]
    x2_data = x2_data * mask_3[2]
    X = x0_data + x1_data + x2_data     # x_data

    y0_data = y0_data * mask_9[0]
    y1_data = y1_data * mask_9[1]
    y2_data = y2_data * mask_9[2]
    Y = y0_data + y1_data + y2_data     # y_data
    
    # for k in range(epochs):
    with hc.loop(0, epochs, step, W) as k:
        wX = W * X
        y_predict = wX + wX.rotate(elements*1) + wX.rotate(elements*2)
        mY = -Y
        error  = y_predict + mY
        error  = error * mask_3[0]
        error0 = X * error
        error1 = X * error.rotate(elements*(16-1))
        error2 = X * error.rotate(elements*(16-2))
        error  = [error0, error1, error2]
        error  = [error[i] + error[i].rotate(elements*(16-1)) for i in range(3)]
        gradW  = [error[i] * hc.Plain([1/2048]) for i in range(3)]
        gradW  = [sum_elements(gradW[i]) for i in range(3)]
        
        concat_gradW  = gradW[0] * mask_3[0]
        concat_gradW += gradW[1] * mask_3[1]
        concat_gradW += gradW[2] * mask_3[2]
        # concat_gradW = concat_gradW * hc.Plain([1/2048])
        
        Wup = concat_gradW * learning_rate
        W = W + Wup
        
        # if a_compile_opt == 16 and (k+1) % 2 == 0 and k != 49:
        #     W = hc.bootstrap(W)
    
    res = [W.rotate(elements*i) for i in range(9)]
     
    return res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7], res[8]


def Multivariate_old (x0_data, x1_data, x2_data, y0_data, y1_data, y2_data) :
    W0 = [hc.Plain([1.0]) for i in range(3)]
    W1 = [hc.Plain([1.5]) for i in range(3)]
    W2 = [hc.Plain([2.0]) for i in range(3)]
    W = [W0, W1, W2]
    X = [x0_data, x1_data, x2_data]
    Y = [y0_data, y1_data, y2_data]
    epochs = 10
    step = 1
    learning_rate = hc.Plain([-0.01])
    inputarr= [W[0][0], W[0][1], W[0][2],W[1][0], W[1][1], W[1][2],W[2][0], W[2][1], W[2][2]]

    for k in range(epochs):
    # with hc.loop(0, epochs, step, W) as k:
        for j in range(3):
            wX = [ X[i] * W[j][i] for i in range(3)]

            y_predict = wX[0] + wX[1] + wX[2]
            mY = -Y[j]
            error0 = y_predict + mY
            error = [ error0 * X[i] for i in range(3)]
            sumerror = [sum_elements(error[i]) for i in range(3)]

            gradW = [sumerror[i] * hc.Plain([1/2048]) for i in range(3)]
            Wup = [gradW[i] * learning_rate for i in range(3)]
            for i in range(3) :
                W[j][i] += Wup[i]
    
    return W[0][0], W[0][1], W[0][2],W[1][0], W[1][1], W[1][2],W[2][0], W[2][1], W[2][2],

modName = hc.save("traced", "traced")
print (modName)
