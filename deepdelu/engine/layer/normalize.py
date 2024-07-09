import numpy as np

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *

class normalize:
    def __init__(self, size:int,
                 Range:tuple[int, int]=[0., 1.]) -> None:
        self.size = size
        self.Range = Range
        self.inclination: float = 1.

        pass

    def compile(self, **kwargs):
        return True
    
    def forward(self, x:np.ndarray):
        self.inclination = (max(self.Range)-min(self.Range))/(max(x)-min(x)+1e-15)
        x = self.inclination*(x - min(x))+min(self.Range)
        return x
    
    def backward(self, **kwargs) -> np.ndarray:

        loss = kwargs.get('loss', 0)
        loss *= self.inclination

        return loss
    
    def update(self, lrate:float) -> None:
        return