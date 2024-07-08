import numpy as np

def MSE(y_true: np.ndarray, y_pred: np.ndarray,derivative:bool=False) -> float:
    '''
    Calculate the Mean Squared Error (MSE) between true and predicted values.

    Parameters:
    y_true (np.ndarray): True values.
    y_pred (numpy.ndarray): Predicted values.

    Returns:
    float: Mean Squared Error.
    '''

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    if (y_true.shape != y_pred.shape):
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    mse: float

    if (not derivative):
        mse = np.mean((y_true - y_pred) ** 2)
    else:
        mse = -2 * (y_true - y_pred) / len(y_true)

    return mse
    
def CategoricalCrossEntropy(y_true:np.ndarray, y_pred:np.ndarray,
                            derivative:bool=False) -> float:
    """
    Calculate the categorical cross-entropy loss.

    Parameters:
    y_true (np.ndarray): True labels, one-hot encoded.
    y_pred (np.ndarray): Predicted probabilities.

    Returns:
    float: Categorical cross-entropy loss.
    """

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if (y_true.shape != y_pred.shape):
        raise ValueError("Shapes of y_true and y_pred must match.")
    
    cce: float

    if (not derivative):
        cce = -1/len(y_true) * np.sum(np.sum(y_true * np.log(y_pred+1e-15)))
    else:
        ...

    return cce