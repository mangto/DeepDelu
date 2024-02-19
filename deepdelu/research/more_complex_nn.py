'''
More-Complex Neural Network With Matrix Calculation
'''

import numpy
from copy import deepcopy

from deepdelu.utils import *

class network:
    def __init__(self, shape:list[int], activation=sigmoid) -> None:
        assert type(shape) == list, "arg 'shape' must be list"
        
        self.shape = shape
        self.acfunc = activation
        
        self.weights = [uniform((s, shape[i])) for i, s in enumerate(shape[1:])]
        self.biases = [uniform(s) for s in shape[1:]]
        
        self.depth = len(shape) - 1
        self.activation :list[numpy.ndarray] # store activation for backpropagation
        self.pure :list[numpy.ndarray] # pure activation
        
        self.description = f"""More-Complex Neural Network\n * shape: {self.shape}"""
        
        pass
    
    def __str__(self) -> str:
        return self.description
    
    def __repr__(self) -> str:
        return self.description
    
    def __call__(self, input:list) -> numpy.ndarray:
        return self.forward(input)
    
    def forward(self, input:list) -> numpy.ndarray:
        '''
        Feed Forward More-Complex Neural Network
        '''
        
        assert len(input) == self.shape[0], "Invalid Input Length"
        
        input = numpy.array(input)
        self.activation = [input]
        self.pure = [input]
        
        for i in range(self.depth):
            
            previous = self.activation[i]
            weights = self.weights[i]
            biases = self.biases[i]
            current = numpy.sum(previous * weights, axis=1) + biases
            
            self.pure.append(current)
            self.activation.append(self.acfunc(current))
            continue
        
        return self.activation[-1]
    
    def backpropagation(self, target:numpy.ndarray, lrate=0.1) -> None:
        '''
        Backpropagation for Two-Depth Neural Network
        '''
        
        assert len(target) == self.shape[-1], "Invalid Target Length"
        target = numpy.array(target)
        
        cost = target - self.activation[-1]
        delta = cost * self.acfunc(self.pure[-1], True)
        weights = deepcopy(self.weights)
        
        self.biases[-1] += delta * lrate
        self.weights[-1] += numpy.outer(delta, self.activation[-2]) * lrate
        
        for i in range(self.depth-1):
            delta = numpy.outer(delta, self.acfunc(self.pure[-i-2], True))
            delta = numpy.sum(delta, axis=0) * weights[-i-1]
            delta = numpy.sum(delta, axis=0)
            
            self.biases[-i-2] += delta * lrate
            self.weights[-i-2] += numpy.outer(delta, self.activation[-i-3]) * lrate
        
        return cost**2/2