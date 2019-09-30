from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)    
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if margin > 0:
                loss += margin
                dW[:,y[i]] -= X[i,:]
                dW[:,j] += X[i,:]   

    # Divide all over training examples
    dW /= num_train
    # Add regularization
    dW += reg * W

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]

    loss = 0.0
    # Calculate the scores matrix
    scores_v = X.dot(W)
    # Calculate the correct score index
    # Create list containing two 1D arrays
    # 1st array = index of images Xi
    # 2nd array = index for correct class value Yi
    correct_score_index = [np.arange(scores_v.shape[0]), y]
    # Select correct score from score matrix using [Xi, Yi]
    correct_class_score_v = scores_v[correct_score_index] 
    # Calculate margin, utilize np.newaxis to broadcast correct class value 
    # across 10 columns
    margin = scores_v - correct_class_score_v[:, np.newaxis] + 1
    # Set values less than zero equal to zero
    margin[margin < 0] = 0
    # Account for condition where j == Yi
    margin[correct_score_index] = 0
    loss = margin.sum()
    loss /= num_train
    loss += 0.5 * reg * np.sum(W * W)

    # This mask can flag the examples in which their margin is greater than 0
    X_mask = np.zeros(margin.shape)
    X_mask[margin > 0] = 1
    
    # As usual, we count the number of these examples where margin > 0
    count = np.sum(X_mask,axis=1)
    X_mask[np.arange(num_train),y] = -count
    
    dW = X.T.dot(X_mask)
    
    # Divide the gradient all over the number of training examples
    dW /= num_train
    
    # Regularize
    dW += reg*W    


    return loss, dW



def svm_loss_simple(W, X, y, reg, debug=False):

    dW = np.zeros(W.shape) # initialize the gradient as zero
    Li = np.zeros(W.shape)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if debug:
                print("Image %s, Class %s, Margin: %s" % (i,j,margin))
            if margin > 0:
                loss += margin
                print(loss)
                dW[:,y[i]] -= X[i,:]
                dW[:,j] += X[i,:]   

    # Divide all over training examples
    dW /= num_train
    # Add regularization
    dW += reg * W

    loss /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)

    
    return loss, dW
