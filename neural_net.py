from __future__ import print_function

from builtins import range
from builtins import object
import numpy as np
import matplotlib.pyplot as plt
from past.builtins import xrange

class TwoLayerNet(object):
    
    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        self.params = {}
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):

        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape

        scores = None

        Z1 = np.dot(X, W1) + b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1,W2) + b2
        scores = Z2

        if y is None:
            return scores

        loss = 0.0
        correctScore = -np.sum(scores[range(N), y])
        loss = correctScore + np.sum(np.log(np.sum(np.exp(scores), axis = 1)),axis=0)
        
        loss /= N
        loss += reg * np.sum(W1* W1) + reg * np.sum(W2 * W2)

        grads = {}

        C = W2.shape[1]
        countOfX = np.zeros((N, C))+ np.exp(scores)/ np.sum(np.exp(scores), axis = 1).reshape(-1,1)
        countOfX[range(N), y] -= 1 
        dZ2 = countOfX
        grads['W2'] = 1/N * np.dot(A1.T, countOfX) + 2 * reg * W2
        grads['b2'] = np.sum(dZ2, axis = 0)/N
        dZ1 = np.dot(dZ2, W2.T)
        dZ1[A1 <= 0] = 0
        grads['W1'] = 1/N * np.dot(X.T, dZ1) + 2 * reg * W1
        grads['b1'] = np.sum(dZ1, axis = 0)/N

        return loss, grads

    def train(self, X, y, X_val, y_val,
              learning_rate=1e-3, learning_rate_decay=0.95,
              reg=5e-6, num_iters=100,
              batch_size=200, verbose=False):

        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):
            X_batch = None
            y_batch = None

            index = np.random.choice(num_train, batch_size,replace = True)
            X_batch = X[index]
            y_batch = y[index]

            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params['W2'] -= learning_rate * grads['W2']
            self.params['b2'] -= learning_rate * grads['b2']
            self.params['W1'] -= learning_rate * grads['W1']
            self.params['b1'] -= learning_rate * grads['b1']


            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            if it % iterations_per_epoch == 0:
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                learning_rate *= learning_rate_decay

        return {
          'loss_history': loss_history,
          'train_acc_history': train_acc_history,
          'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        Z1 = np.dot(X, self.params['W1']) + self.params['b1']
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(A1,self.params['W2']) + self.params['b2']
        y_pred = np.argmax(Z2, axis = 1)

        return y_pred
