from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

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
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # f(X[i],W) = W*X[i] --> f_j = W[:,j]*X[i] --> df_j = X[i]
    # normprob = e^f_y[i] / sum_j {e^f_j}
    # loss = (1/num_train) * -log(normprob) + reg * |W|^2 = (1/num_train) * (-f_y[i] + log(sum_j {e^f_j})) + reg * |W|^2
    # dW_k = -X[i] (if k == y[i]) + 1/(sum_j {e^f_j}) * e^f_k * df_k 
    # ---> = -X[i] (if k == y[i]) + e^f_k * X[i]/(sum_j {e^f_j}) 
    # dW = (1/num_train) * [dW_k] + 2* reg * W
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W) # f(X[i],W) = W*X[i]
        exp_scores = np.exp(scores) # e^f
        exp_corr_score = exp_scores[y[i]] # e^f_y[i]
        normprob = exp_corr_score / np.sum(exp_scores) # e^f_y[i] / sum_j {(e^(f_j))}
        loss -= np.log(normprob) # loss[i] = -log(norm_prob)
        for k in range(num_classes):
            # dW_k = -X[i] (if k == y[i]) + e^f_k * X[i]/(sum_j {e^f_j})
            dW[:,k] += (-X[i] * (k == y[i]) + exp_scores[k] * X[i] / np.sum(exp_scores)) 

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train 

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    dW += 2 * reg * W 

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    # NAIVE
    # f(X[i],W) = W*X[i] --> f_j = W[:,j]*X[i] --> df_j = X[i]
    # normprob = e^f_y[i] / sum_j {e^f_j}
    # loss = (1/num_train) * -log(normprob) + reg * |W|^2 = (1/num_train) * (-f_y[i] + log(sum_j {e^f_j})) + reg * |W|^2
    # dW_k = -X[i] (if k == y[i]) + 1/(sum_j {e^f_j}) * e^f_k * df_k 
    # ---> = -X[i] (if k == y[i]) + e^f_k * X[i]/(sum_j {e^f_j}) 
    # dW = (1/num_train) * [dW_k] + 2* reg * W
    
    # VECTORIZED
    # F(X,W) = X*W --> dF = X
    # Q = (e^F / sum {e^F}) 
    # NormProb = Q[:,y] 
    # loss = (1/num_train) * sum(-log(NormProb)) + reg * |W|^2
    # dW = (1/num_train) * -1/NormProb * dNP + 2 * reg * W = (1/num_train) * -dNP/NormProb + 2 * reg * W
    # -dNP/NormProb = X^T * (Q - V), where V_ik = 1 if k == y[i] (else V_ik = 0)
    
    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    scores = X.dot(W) # F(X,W) = X*W
    exp_scores = np.exp(scores) # e^F
    exp_scores_sum = np.sum(exp_scores, axis = 1).reshape(num_train, 1) 
    Q = exp_scores / exp_scores_sum
    NormProb = Q[np.arange(num_train), y] # NormProb = Q[:,y]
    
    loss = np.sum(-np.log(NormProb))/num_train + reg * np.sum(W*W)
    V = np.zeros_like(Q)
    V[np.arange(num_train), y] = 1
    dW = (X.T).dot(Q - V)/num_train + 2 * reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
