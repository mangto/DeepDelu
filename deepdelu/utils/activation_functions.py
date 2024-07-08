import numpy

# Activation Functions
def sigmoid(value, derivative=False):
    value = numpy.array(value)

    if (derivative == False): return 1/(1+numpy.exp(-1*value))
    else:
        sig = sigmoid(value)
        return sig*(1-sig)

def relu(value, derivative=False):
    value = numpy.array(value)

    if (derivative == False): return numpy.maximum(0, value)
    else:
        value[value<=0] = 0
        value[value>0] = 1
        return value

def leaky_relu(value, derivative=False):
    value = numpy.array(value)

    if (derivative == False): return numpy.maximum(0.01*value, value)
    else:
        value = numpy.array(value)
        value[value<=0] = 0.01
        value[value>0] = 1
        return value

def linear(value, derivative=False):
    value = numpy.array(value)

    if (derivative == False): return value
    else: return numpy.ones(value.shape)

def tanh(value, derivative=False):
    value = numpy.array(value)

    if (derivative == False): return numpy.tanh(value)
    else: return 1/numpy.cosh(value)
    
def softmax(value, derivative = False):
    if (derivative == False): e_x = numpy.exp(value); return e_x / e_x.sum()
    else: s = softmax(value).reshape(-1,1); return numpy.diag(numpy.diagflat(s) - numpy.dot(s, s.T))