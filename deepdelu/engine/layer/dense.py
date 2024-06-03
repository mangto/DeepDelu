import numpy

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *

class dense:
    def __init__(self, size:int, activation=sigmoid, InputShape:int=None) -> None:
        
        self.size = size
        self.activation = activation
        self.InputShape = InputShape
        
        pass
    
    def do(self) -> numpy.ndarray:
        return
    
    def compile(self) -> numpy.ndarray:
        return