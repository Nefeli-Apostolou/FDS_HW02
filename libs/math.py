import numpy as np


def sigmoid(x):
    """
    Function to compute the sigmoid of a given input x.

    Args:
        x: it's the input data matrix.

    Returns:
        g: The sigmoid of the input x
    """
    ##############################
    ###     YOUR CODE HERE     ###
    g = 1/ (1 + np.exp(-x))
    ##############################    
    return g

def softmax(y):
    """
    Function to compute associated probability for each sample and each class.

    Args:
        y: the predicted 

    Returns:
        softmax_scores: it's the matrix containing probability for each sample and each class. The shape is (N, K)
    """
    ##############################
    ###     YOUR CODE HERE     ###
    # Shift the scores for numerical stability (subtract the max score in each row)
    y_shift = y - np.max(y, axis=1, keepdims=True)
    
    # Compute the exponential of the shifted scores
    exp_scores = np.exp(y_shift)
    
    # Normalize by dividing each row by the sum of its exponential scores
    softmax_scores = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    ##############################
    return softmax_scores

