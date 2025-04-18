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

def cross_entropy(y_true: ValueTensor, y_pred: ValueTensor):
    """y_true: one‑hot, y_pred: raw logits."""
    # 1) compute softmax
    probs = y_pred.softmax()                                    # shape (batch, C)

    # 2) compute -∑ y_true·log(probs) averaged over batch
    #    we need a log; if not implemented, add to ValueTensor
    log_probs = probs.log()                                     # you’ll have to implement log()
    loss = -(y_true * log_probs).sum(axis=1, keepdims=True)     # shape (batch, 1)
    return loss.mean(axis=0, keepdims=False)                    # scalar
