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
                dW[:,y[i]] -= X[i] # ADDED CODE
                dW[:,j] += X[i] # ADDED CODE

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train # ADDED CODE

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W # ADDED CODE

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # delta = 1
    # f = f(X[i], W) = WX[i] -----> f_j = W[:,j]*X[i] -----> df_j = X[i]
    # loss = (1/num_train) * sum{ max(0, f_j - f_y[i] + delta)) } + reg * |W|^2
    # dW = grad of loss = (1/num_train) * sum_(j != y[i]){ max(0, df_j - df_y[i]) } + 2 * reg * W
    # ADDED CODE above
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # NAIVE
    # delta = 1
    # f = f(X[i], W) = WX[i] -----> f_j = W[:,j]*X[i] -----> df_j = X[i]
    # loss = (1/num_train) * sum{ max(0, f_j - f_y[i] + delta)) } + reg * |W|^2
    # dW = grad of loss = (1/num_train) * sum_(j != y[i]){ max(0, df_j - df_y[i]) } + 2 * reg * W
    
    # VECTORIZED
    # delta = 1
    # F = F(X,W) = [f] = X*W -----> dF = X
    # F_correct = F(when F has label y) = [F(i, y[i])]
    # loss = (1/num_train) * max_(F != F_correct){ (0, F - F_correct + delta) } + reg * |W|^2
    # dW = grad of loss = (1/num_train) * max_([j] != y){0, dF - dF_correct} + 2 * reg * W
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W) # F(X,W)
    correct_class_score = scores[np.arange(num_train), y].reshape(num_train, 1) # F_correct
    margin = np.maximum(0, scores - correct_class_score + 1)
    margin[np.arange(num_train), y] = 0 # when scores match up with correct_class_score (i.e., F == F_correct)
    loss = margin.sum() / num_train + reg * np.sum(W*W)
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    # VECTORIZED
    # delta = 1
    # F = F(X,W) = [f] = X*W -----> dF = X
    # F_correct = F(when F has label y) = [F(i, y[i])] -----> dF_correct = X when correct
    # loss = (1/num_train) * max_(F != F_correct){ (0, F - F_correct + delta) } + reg * |W|^2
    # dW = grad of loss = (1/num_train) * max_([j] != y){0, dF - dF_correct} + 2 * reg * W
    
    margin[margin > 0] = 1 # errors set to 1, correct classifs left at 0
    # sum(axis = 1) -----> add across rows
    # for each row, sum all errors and subtract total error from 0 where correct entries are
    
    # CASE 1:
    # If a row of 4 classes (columns) incorrectly classifies class 3 as class 2, then margin is set to 1,1,1,1.
    # Then total_error = 4. If we subtract total_error from correct class, then we get V=[1,1,1,-3].
    
    # CASE 2:
    # If a row of 4 classes correctly classifies class 3, then margin is set to 1,1,1,0.
    # Then total_error = 3. If we subtract total_error from correct class, then we get V=[1,1,1,-3].
    
    margin[np.arange(num_train), y] -= margin.sum(axis = 1) # prep V for dF - dF_correct = X^T * V
    dW = (X.T).dot(margin) / num_train + 2 * reg * W
    
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
