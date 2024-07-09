import numpy as np

class flatten:
    def __init__(self, size:tuple[int, int]=None) -> None:
        self.size = size

        self.size = np.prod(size)
        
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