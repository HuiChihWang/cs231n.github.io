import numpy as np
from random import shuffle

def softmax(x):
    shiftx = x - np.max(x)
    exps = np.exp(shiftx)
    return exps / (np.sum(exps)+1e-5)


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
  train_num = X.shape[0]
  class_num = W.shape[1]
  for i in range(train_num):
    score = X[i] @ W
    softmax_score = softmax(score)
    loss -= np.log(softmax_score[y[i]])
    
    delta_softmax = -softmax_score * softmax_score[y[i]]
    delta_softmax[y[i]] += softmax_score[y[i]]
    
    for j in range(class_num):
        dW[:,j] += softmax_score[j] * X[i,:]
        if y[i] == j:
            dW[:,j] -= X[i,:]
    
  loss /= train_num
  loss +=  reg *np.sum(W*W)
    
  dW /= train_num
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

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
  train_num = X.shape[0]
  scores = X @ W
  shift_score = scores - np.amax(scores,axis=1,keepdims=True)
  exps = np.exp(shift_score)
  softmax_scores = exps / (np.sum(exps,axis=1,keepdims=True)+1e-5)
  
  log_loss = -np.log(softmax_scores[np.arange(train_num),y])
  loss = np.mean(log_loss)
  loss += reg *np.sum(W*W)

  dloss = softmax_scores
  dloss[np.arange(train_num),y] -= 1
  dW = X.T @ dloss
  
  dW /= train_num
  dW += 2*reg*W
  
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

