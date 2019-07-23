from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros(W.shape)

    num_train = X.shape[0]
    scores = np.dot(X,W)
    m = np.maximum(0, scores - scores[range(num_train), y].reshape(-1, 1) + 1.0)
    m[range(num_train), y] = 0
    loss = np.sum(m)/num_train + reg * np.sum(W * W)

    num_class = W.shape[1]
    countOfXT = np.zeros((num_train, num_class))
    countOfXT[m>0]=1
    countOfXT[range(num_train), y]=-np.sum(countOfXT, axis = 1)
    
    dW = np.dot(X.T, countOfXT)/num_train + 2*reg*W

    return loss, dW
