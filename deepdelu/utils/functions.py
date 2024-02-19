import numpy

def uniform(shape:tuple) -> numpy.ndarray:
    return numpy.random.uniform(-1.0, 1.0, shape)