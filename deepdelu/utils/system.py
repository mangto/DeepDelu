import os

def clear():
    '''
    clear console
    '''
    os.system("cls")

def division(length:int) -> None:
    '''
    Simple division for better console

    Parameters:
    length (int): length of division

    Returns:
    None

    '''
    assert type(length) == int, "Invalid Length"

    length = max(length, 5)

    print("+" + "="*(length-2) + "+")

    return