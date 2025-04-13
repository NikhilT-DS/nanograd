import numpy as np
from engine import ValueTensor

def mse(y, y_pred):
    """ implementation of Mean Squared Error loss function
    Inputs: expects the actual target and predicted target
    Outputs: returns the MSE value
    """
    return ((y - y_pred) ** 2).mean()

# TODO: write the functions.
def binary_cross_entropy(y, y_pred):
    pass

def hinge_loss():
    pass

def cross_entropy():
    pass