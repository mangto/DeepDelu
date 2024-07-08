import numpy as np

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *

class flatten:
    def __init__(self, InputShape:tuple[int, int]=None) -> None:
        self.InputShape = InputShape

        self.size = np.prod(InputShape)
        
        pass

    def compile(self, **kwargs):
        return True
    
    def forward(self, x:np.ndarray):
        return x.flatten()
    
    def backward(self, **kwargs) -> np.ndarray:

        loss = kwargs.get('loss', 0)
        return loss
    
    def update(self, lrate:float) -> None:
        return