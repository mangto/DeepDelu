import numpy as np
from tqdm import tqdm

from deepdelu.utils.activation_functions import *
from deepdelu.utils.functions import *
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
        self.optimizer = None
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

    def compile(self, loss=MSE, optimizer=None) -> bool:
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
            self.optimizer = optimizer

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

    def forward(self, x:np.ndarray) -> np.ndarray:
        '''
        Feed Forward Network.

        Parameters:
        x (ndarray): input value.

        Returns:
        ndarray: output of network.
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
        x (ndarray): input value.

        Returns:
        ndarray: output of network.
        '''
        return self.forward(x)

    def backward(self, y_true:np.ndarray, y_pred:np.ndarray) -> None:
        '''
        Backpropagation Process.

        Parameters:
        y_true (ndarray): answer
        y_pred (ndarray): prediction

        Returns:
        None
        '''
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

    def update(self) -> None:
        '''
        Update Network.
        You must run this process to train network.
        '''
        assert self.compiled, "Network must be compied before using"
        
        self.lrate = self.lrate if not self.optimizer else self.optimizer()

        layer: dense
        for layer in self.layers:
            layer.update(self.lrate)
    
    def compute_loss(self, y_true:np.ndarray, y_pred:np.ndarray,
                     derivative:bool=False
                     ) -> float:
        '''
        Compute loss with own loss function.
        Default loss functions is MSE.
        You can change it when .compile()

        Parameters:
        y_true  (ndarray): answer
        y_pred  (ndarray): prediction
        derivative (bool): whether compute derivative or not.

        Returns:
        float|ndarray: computed result.

        '''
        assert self.compiled, "Network must be compied before using"

        loss = self.loss(y_true, y_pred, derivative)
        
        return loss

    def train(self,
              x_train:np.ndarray, y_train:np.ndarray, epoch:int,
              ShowLoss:bool=True,
              SaveModel:bool=True, ModelPath:str=".\\model.pkl"
              ) -> None:
        '''
        Train network

        Parameters:
        x_train (ndarray): train input
        y_train (ndarray): answer
        ShowLoss   (bool): whether to show loss or not.
        SaveModel  (bool): whether to save model or not.
        ModelPath   (str): path to save model.

        Returns:
        None
        '''
        assert self.compiled, "Network must be compied before using"
        assert len(x_train) == len(y_train), "Length of x_train and y_train must be same"
        assert type(epoch) == int, "Invalid epoch size"

        count = len(x_train)

        for epc in range(epoch):

            loss = 0
            
            for i in tqdm(range(count)):

                X, Y = x_train[i], y_train[i]

                pred = self.forward(X)
                self.backward(Y, pred)
                self.update()

                loss += self.compute_loss(Y, pred)

                continue

            if(ShowLoss): print(f"epoch: {epc}    loss: {loss}")
            if(SaveModel): save_model(self, ModelPath)
            continue

        return
    
    def evaluate(self,
                 x_test:np.ndarray, y_test:np.ndarray,
                 ShowResult: bool = True
                 ) -> tuple[float, float]:
        '''
        Evaluate model with test datasets

        Parameters:
        x_test  (ndarray): test input
        y_test  (ndarray): answer
        ShowResult (bool): whether to show result or not

        Returns:
        tupele[float, float]: loss, accuracy
        '''

        assert self.compiled, "Network must be compied before using"

        count = len(y_test)
        loss = 0
        correct = 0
        for i in tqdm(range(count)):
            X, Y = x_test[i], y_test[i]

            pred = self.forward(X)
            loss += self.compute_loss(Y, pred)
            correct += pred.argmax() == Y.argmax()

        accuracy = correct/count

        if (ShowResult): print(f"loss: {loss}    accuracy: {round(accuracy*100, 3)}% ({correct}/{count})")
        
        return loss, accuracy