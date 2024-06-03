import numpy

from deepdelu.utils.activation_functions import *
from deepdelu.engine.tensor import Tensor

class network:
    def __init__(self, lrate:float=0.1) -> None:

        self.lrate: float = lrate
        self.layers: list = []

        pass