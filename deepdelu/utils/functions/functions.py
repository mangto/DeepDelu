import numpy
from pickle import dump, load, HIGHEST_PROTOCOL
from os.path import isfile

def uniform(shape:tuple) -> numpy.ndarray:
    return numpy.random.uniform(-1.0, 1.0, shape)

def save_model(model:object, path=".\\model.pkl") -> None:
    with open(path, 'wb') as file:
        dump(model, file, protocol=HIGHEST_PROTOCOL)
    return

def load_model(path):
    if (not isfile(path)): return
    
    with open(path, "rb") as file:
        model = load(file)
    return model