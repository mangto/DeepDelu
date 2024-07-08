import numpy as np

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *

class flatten:
    def __init__(self, InputShape:int=None) -> None:
        self.InputShape = InputShape
        
        pass

    def compile(self, **kwargs):
        return True
    
    def forward(self):
        return
    
    def backward(self):
        return
    