'''
Two-Depth Neural Network With Matrix Calculation
'''

import numpy as np
from copy import deepcopy

from deepdelu.utils import *

class network:
    def __init__(self, shape:list[int, int, int], activation=sigmoid) -> None:
        assert type(shape) == list, "arg 'shape' must be list"
        assert len(shape) == 3, "arg 'shape' must contain counts of neurons in Input, Layer0, Output"
        
        self.shape = shape
        self.acfunc = activation
        
        self.weights = [np.random.uniform(-1/s, 1/s, (s, shape[i])) for i, s in enumerate(shape[1:])]
        self.biases = [np.zeros((s, )) for i, s in enumerate(shape[1:])]
        
        self.depth = len(shape) - 1
        self.activation :list[np.ndarray] # store activation for backpropagation
        self.pure :list[np.ndarray] # pure activation
        
        self.description = f"""Two-Depth Neural Network\n * shape: {self.shape}"""
        
        pass
    
    def __str__(self) -> str:
        return self.description
    
    def __repr__(self) -> str:
        return self.description
    
    def __call__(self, input:list) -> np.ndarray:
        return self.forward(input)
    
    def forward(self, input:list) -> np.ndarray:
        '''
        Feed Forward Two-Depth Neural Network
        '''
        
        assert len(input) == self.shape[0], "Invalid Input Length"
        
        input = np.array(input)
        self.activation = [input]
        self.pure = [input]
        
        for i in range(self.depth):
            
            previous = self.activation[i]
            weights = self.weights[i]
            biases = self.biases[i]
            current = np.sum(previous * weights, axis=1) + biases
            
            self.pure.append(current)
            self.activation.append(self.acfunc(current))
            continue
        
        return self.activation[-1]
    
    def backpropagation(self, target:np.ndarray, lrate=0.1) -> None:
        '''
        Backpropagation for Two-Depth Neural Network
        '''
        
        assert len(target) == self.shape[-1], "Invalid Target Length"
        target = np.array(target)
        
        cost = target - self.activation[-1]
        delta = cost * self.acfunc(self.pure[-1], True)
        weights = deepcopy(self.weights)
        
        self.biases[-1] += delta * lrate
        self.weights[-1] += np.outer(delta, self.activation[-2]) * lrate
        
        delta = np.outer(delta, self.acfunc(self.pure[-2], True))
        delta = np.sum(delta, axis=0) * weights[-1]
        delta = np.sum(delta, axis=0)
        
        self.biases[-2] += delta * lrate
        self.weights[-2] += np.outer(delta, self.activation[-3]) * lrate
        
        return cost**2/2