import numpy as np
from deepdelu.engine.tensor import Tensor

def norm(array: np.ndarray, p: float = 2.) -> float:
    '''
    Lp Norm
     * array: tuple | list | np.array | np.ndarray | Tensor
     * p    : float | int
    '''

    assert type(array) in (tuple, list, np.array, np.ndarray, Tensor), 'Invalid Type of Array'
    assert type(p) in (float, int), 'Invalid p Value, It must be integer or float'

    if (p == float('inf')): return Tensor([max(array)]) # L-infinity norm

    result: float = sum([abs(val)**p for val in array]) ** (1/p)

    return Tensor([result])

def cosine_similiarity(arr1: np.ndarray, arr2: np.ndarray, p: float = 2.) -> float:

    assert type(arr1) in (tuple, list, np.array, np.ndarray, Tensor), 'Invalid Type of arr1'
    assert type(arr2) in (tuple, list, np.array, np.ndarray, Tensor), 'Invalid Type of arr2'
    assert type(p) in (float, int), 'Invalid p Value, It must be integer or float'

    if (type(arr1) == Tensor): arr1 = arr1.data
    if (type(arr2) == Tensor): arr2 = arr2.data

    arr1, arr2 = np.array(arr1), np.array(arr2)

    return (arr1 @ arr2) / (norm(arr1, p).data[0] * norm(arr2, p).data[0])