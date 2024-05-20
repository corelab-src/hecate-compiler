import hecate as hc
import sys


def sum_elements(data):
    for i in range(12):
    # with hc.loop(0, 12, 1, [data]) as i:
        rot = data.rotate(1<<(11-i))
        # rot = data.rotate(i)
        data = data +rot

    return data

def create_mask(elements):
    mask = [[] for i in range(3)]

    temp0 = [0.0 for i in range(elements)]
    temp1 = [1.0 for i in range(elements)]

    mask[0] = temp1 + temp0 
    mask[1] = temp0 + temp1
    mask[2] = temp1 + temp1

    mask[0] = hc.Plain(mask[0])
    mask[1] = hc.Plain(mask[1])
    mask[2] = hc.Plain(mask[2])
    
    return mask

@hc.func("c,c")
def LinearRegression(x_data, y_data) :
    
    epochs = 10
    step = 1
    learning_rate = hc.Plain([-0.01])
    
    elements = 4096
    mask = create_mask(elements)
    W = mask[1] 
    
    for i in range(epochs):
    # with hc.loop(0, epochs, step, inputarr = W) as i:
        
        xW = x_data * W
        y_predict = W + xW.rotate(elements)
        mY = - y_data

        error0 = y_predict + mY
        error0 = error0 * mask[0]
        error0 = error0 + error0.rotate(elements*(16-1)) 
        error1 = error0 * x_data
        error  = [error0, error1]
        gradW  = [error[i] * hc.Plain([1/2048]) for i in range(2)]
        gradW  = [sum_elements(gradW[i]) for i in range(2)]

        concat_gradW  = gradW[0] * mask[0]
        concat_gradW1 = gradW[1] * mask[0]
        concat_gradW += concat_gradW1.rotate(elements*(16-1)) 
        # concat_gradW = concat_gradW * hc.Plain([1/2048])

        Wup = concat_gradW * learning_rate
        W = W + Wup

        # if a_compile_opt == 16 and (i+1) % 2 == 0 and i != 49:
        #     W = hc.bootstrap(W)

    res = [W.rotate(elements*i) for i in range(2)]

    return res[1], res[0]

def LinearRegression_old(x_data, y_data) :
    W = hc.Plain([1.0])
    b = hc.Plain([0.0])
    
    epochs = 1
    learning_rate = hc.Plain([-0.01])

    # inputarr = [hc.bootstrap(W), hc.bootstrap(b)]
    step = 1
    # with hc.loop(0, epochs, step,  init_vars = (inputarr, W, b)) as loop:
    with hc.loop(0, epochs, step, inputarr = [W,b]) as i:
    # for k in range(epochs):
        # t = x_data +i 
        xW = x_data*W
        xWb = xW + b
        error = xWb - y_data
        errX = error * x_data
        meanErrX = errX * hc.Plain([1/2048])
        gradW = sum_elements(meanErrX)
        meanErr = error * hc.Plain([1/2048])
        gradb = sum_elements(meanErr)
        Wup = learning_rate * gradW
        bup = learning_rate * gradb
        W = W + Wup
        b = b + bup
        # W = hc.bootstrap(W)
        # b = hc.bootstrap(b)

    return W, b



modName = hc.save("traced", "traced")
print (modName)


