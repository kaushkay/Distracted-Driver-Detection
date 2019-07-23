from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):

    loss = 0.0
    dW = np.zeros_like(W)

    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        score = np.dot(X[i].T, W)
        correctScore = score[y[i]]
        loss -= np.log(np.exp(correctScore)/np.sum(np.exp(score)))
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += - X[i] + np.exp(score[j])/ np.sum(np.exp(score)) * X[i] 
            else:
                dW[:, j] += np.exp(score[j])/ np.sum(np.exp(score)) * X[i]
    loss /= num_train 
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2* reg * W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):

    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]
    score = np.dot(X, W)
    correctScore = -np.sum(score[range(num_train), y])
    loss = correctScore + np.sum(np.log(np.sum(np.exp(score), axis = 1)),axis=0)
    loss /= num_train
    loss += reg * np.sum(W * W)
    
    num_classes = W.shape[1]
    countOfX = np.zeros((num_train, num_classes))+ np.exp(score)/ np.sum(np.exp(score), axis = 1).reshape(-1,1)
    countOfX[range(num_train), y] -= 1 
    dW = np.dot(X.T, countOfX)
    dW /= num_train
    dW += 2 * reg * W

    return loss, dW
