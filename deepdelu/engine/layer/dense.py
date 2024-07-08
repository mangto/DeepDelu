import numpy as np

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *

class dense:
    def __init__(self, size:int, acfunc=sigmoid, InputShape:int=None) -> None:
        
        self.size: int = size
        self.acfunc: sigmoid = acfunc
        self.InputShape: int = InputShape

        self.weights: np.ndarray
        self.biases: np.ndarray
        self.raw: np.ndarray
        self.activation: np.ndarray
        self.previous: np.ndarray

        self.dweights: np.ndarray
        self.dbiases: np.ndarray
        
        pass

    def compile(self, **kwargs):

        previous_size: int
        previous_layer = kwargs.get('previous_layer', None)
        

        if (previous_layer): previous_size = previous_layer.size
        elif (self.InputShape): previous_size = self.InputShape
        else:
            raise Exception(
                """Invalid Previous-layer was given, maybe you have to set InputShape"""
                )
        
        shape: tuple = (self.size, previous_size)
        RANGE: float = 1/previous_size
        
        self.weights = np.random.uniform( -1 * RANGE, RANGE, shape )
        self.biases = np.zeros((self.size, ))

        return True
    
    def forward(self, x:np.ndarray):
        self.previous = x                           # save previous activation for later calculation
        self.raw = self.weights @ x + self.biases   # not-acfunctioned activation
        self.activation = self.acfunc( self.raw )   # activation

        return self.activation
    
    def backward(self, **kwargs) -> np.ndarray:

        loss = kwargs.get('loss', 0)
        grad = loss * self.acfunc(self.raw, True)

        self.dweights = grad[:, np.newaxis] * self.previous
        self.dbiases = grad

        grad = np.sum(grad * self.weights.T, axis=1)

        return grad
    
    def update(self, lrate:float) -> None:

        self.weights -= lrate * self.dweights
        self.biases -= lrate * self.dbiases

        return