from random import random

class network:
    
    def __init__(self) -> None:
        '''
        Initalize Linear-Equation Neural Network
        The Most Simplest Neural Network

        INPUT  ->  OUTPUT
              ^^^^
          Single Neuron

        f(x) = ax + b
        '''
        
         # random() returns float between 0 and 1. Therefore, we have to adjust it between -1 and 1
        self.weight :float = random()*2-1
        self.bias = 0 # bias normally begins with 0
        
        pass
    
    
    def __call__(self, x:int|float) -> float:
        return self.forward(x)
    
    def __str__(self) -> str:
        return f"f(x) = {round(self.weight, 2)}x + {round(self.bias, 2)}"
    
    def __repr__(self) -> str:
        return f"f(x) = {round(self.weight, 2)}x + {round(self.bias, 2)}"
    
    def forward(self, x:int|float) -> float:
        '''
        Feed Forward Linear-Equation Neuron Neural Network
        '''
        
        assert type(x) in [int, float], "arg 'x' must be int or float"
        
        return self.weight * x + self.bias
    
    def backpropagation(self, x:int|float, target:int|float, lrate:float=0.01) -> float:
        '''
        Train Network With Backpropagation
        '''
        assert type(x) in [int, float], "arg 'x' must be int or float"
        assert type(target) in [int, float], "arg 'target' must be int or float"
        
        o = self.forward(x) # Output
        dc = target - o # Delta Cost
        self.weight += dc * x * lrate
        self.bias += dc * lrate
        
        return dc