import numpy as np

from deepdelu.utils.activation_functions import *
from deepdelu.utils.losses import *
from deepdelu.utils.system import *
from deepdelu.engine.layer import *

class network:

    def __init__(self, lrate:float=0.1) -> None:
        '''
        Initalize Network.

        Parameters:
        lrate (float) = 0.1

        Returns:
        None
        '''
        self.lrate: float = lrate

        self.layers: list = []
        self.compiled: bool = False
        self.loss = MSE
        self.activation = []

    def add_layer(self, *layer:dense) -> None:
        '''
        Add Layer to Network.
        After you add all layers, YOU MUST COMPILE NETWORK.

        Parameters:
        layer (deepdelu.layer.dense): network layer.

        Returns:
        None
        '''
        for lay in layer:
            if (type(lay) in (list, tuple)):
                self.layers += list(lay)
                continue
            self.layers.append(lay)

    def compile(self, loss=MSE) -> bool:
        '''
        Compile Network.
        This function initalizes layers' weights and biases.
        Therefore, you must compile network to use it.

        Parameters:
        loss (function): loss function like MSE.

        Returns:
        bool (True if succeed, False else)
        '''
        try:
            
            self.loss = loss

            layer: dense
            previous_layer: dense = None

            for layer in self.layers:
                layer.compile(
                    previous_layer=previous_layer
                    )
                previous_layer = layer
                continue

            self.compiled = True

            return True

        except Exception as e:
            # When whatever exception occurs
            print(e)
            return False

    def forward(self, x:np.ndarray):
        '''
        Feed Forward Network.

        Parameters:
        x (np.ndarray): input value.

        Returns:
        np.ndarray: output of network.
        '''
        assert self.compiled, "Network must be compied before using"

        self.activation = [np.array(x)]
        layer: dense

        for layer in self.layers:
            activation = layer.forward(self.activation[-1])
            self.activation.append(activation)
            continue

        return self.activation[-1]
    
    def __call__(self, x: np.ndarray) -> np.ndarray:
        '''
        Feed Forward Network.

        Parameters:
        x (np.ndarray): input value.

        Returns:
        np.ndarray: output of network.
        '''
        return self.forward(x)

    def backward(self, y_true:np.ndarray, y_pred:np.ndarray) -> None:
        assert self.compiled, "Network must be compied before using"
        
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        if (y_true.shape != y_pred.shape):
            raise ValueError("Shapes of y_true and y_pred must match.")
        
        loss = self.compute_loss(y_true, y_pred, True) # calculate derivative loss
        
        layer: dense # for easier programming

        for layer in self.layers[::-1]:
            loss = layer.backward(
                loss = loss
            )
            continue


    def update(self):
        assert self.compiled, "Network must be compied before using"
        
        layer: dense
        for layer in self.layers:
            layer.update(self.lrate)
    
    def compute_loss(self, y_true:np.ndarray, y_pred:np.ndarray,
                     derivative:bool=False
                     ) -> float:
        assert self.compiled, "Network must be compied before using"

        loss = self.loss(y_true, y_pred, derivative)
        
        return loss

    def train(self):
        assert self.compiled, "Network must be compied before using"
        ...
    
    def predict(self):
        assert self.compiled, "Network must be compied before using"
        ...
    
    def evaluate(self):
        assert self.compiled, "Network must be compied before using"
        ...