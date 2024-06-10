import hecate as hc
import sys

if(len(sys.argv) != 1):
    a_epochs = int(sys.argv[1])

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
    epochs = a_epochs
    W = hc.Plain([1.0])
    b = hc.Plain([0.0])
    
    learning_rate = hc.Plain([-0.01])

    # inputarr = [hc.bootstrap(W), hc.bootstrap(b)]
    for k in range(epochs):
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


