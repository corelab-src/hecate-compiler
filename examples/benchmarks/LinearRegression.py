import hecate as hc
import sys


def sum_elements(data):
    # for i in range(12):
    for i in range(12):
    # with hc.loop(0, 12, 1) as i:
        rot = data.rotate(1<<(11-i))
        data = data +rot

    return data

@hc.func("c,c")
def LinearRegression(x_data, y_data) :
    W = hc.Plain([1.0])
    b = hc.Plain([0.0])
    
    epochs = 2
    learning_rate = hc.Plain([-0.01])

    # for k in range(epochs):
    # with hc.loop(0, epochs, step) as i: 
    zero = x_data - x_data
    # inputarr = [W+zero, b+zero]
    inputarr = [W, b]
    step = 1
    # with hc.loop(0, epochs, step,  init_vars = (inputarr, W, b)) as loop:
    # with hc.loop(0, epochs, step, inputarr) as loop:
    # with hc.loop(0, epochs, step, inputarr) as i:
    # with hc.loop(0, epochs, step) as i:
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

    W = hc.bootstrap(W)
    b = hc.bootstrap(b)

    return W, b



modName = hc.save("traced", "traced")
print (modName)


