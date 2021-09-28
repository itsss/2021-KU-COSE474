'''
HW2 problem
'''

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import data_generator as dg
from collections import Counter

# you can define/use whatever functions to implememt

########################################
# Part 1. cross entropy loss
########################################
def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))

    scores = W@x+b
    loss = 0.0
    reg = 1.0

    num_train = x.shape[1]

    ce = np.exp(scores)
    ce /= np.sum(ce, axis=0, keepdims=True)
    loss -= np.sum(np.log(ce[y, np.arange(num_train)]))

    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    return loss

########################################
# Part 2. SVM loss calculation
########################################
def svm_loss(Wb, x, y, num_class, n, feat_dim):
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * feat_dim)], (num_class, feat_dim))
    x = np.reshape(x.T, (-1, n))

    scores = W@x+b
    loss = 0.0
    reg = 1.0

    num_train = x.shape[1]

    seq = np.array(range(num_train))
    correct = scores[y, seq]
    margin = np.maximum(0, scores - correct + 1)
    margin[y, seq] = 0
    loss = np.sum(margin)

    loss /= num_train
    loss += 0.5 * reg * np.sum(W ** 2)

    return loss

########################################
# Part 3. kNN classification
########################################
class kNN:
    def __init__(self, num_n=3):
        self.k = num_n

    def predict(self, X_train, y_train, X_test):
        self.points = X_train
        self.labels = y_train
        predicts = []
        for test_pt in X_test:
            dist = np.sqrt(np.sum((self.points - test_pt) ** 2, axis=1))
            winner = self.classify(dist)
            predicts.append(winner)
        return predicts

    def classify(self, dist):
        dist_val = np.argsort(dist)
        knn_neigh = []
        for i in dist_val[0:self.k]:
            knn_neigh.append(self.labels[i])
        cnt = Counter(knn_neigh)
        win, win_cnt = cnt.most_common(1)[0]
        return win

def knn_test(X_train, y_train, X_test, y_test, n_train, n_test, k):
    knn = kNN(num_n=k)
    y_pred = knn.predict(X_train, y_train, X_test)
    return np.mean(y_pred == y_test)

# now lets test the model for linear models, that is, SVM and softmax
def linear_classifier_test(Wb, x_te, y_te, num_class,n_test):
    Wb = np.reshape(Wb, (-1, 1))
    dlen = len(x_te[0])
    b = Wb[-num_class:]
    W = np.reshape(Wb[range(num_class * dlen)], (num_class, dlen))
    accuracy = 0

    for i in range(n_test):
        # find the linear scores
        s = W @ x_te[i].reshape((-1, 1)) + b
        # find the maximum score index
        res = np.argmax(s)
        accuracy = accuracy + (res == y_te[i]).astype('uint8')

    return accuracy / n_test

# number of classes: this can be either 3 or 4
num_class = 4

# sigma controls the degree of data scattering. Larger sigma gives larger scatter
# default is 1.0. Accuracy becomes lower with larger sigma
sigma = 1.0

print('number of classes: ',num_class,' sigma for data scatter:',sigma)
if num_class == 4:
    n_train = 400
    n_test = 100
    feat_dim = 2
else:  # then 3
    n_train = 300
    n_test = 60
    feat_dim = 2

# generate train dataset
print('generating training data')
x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

# generate test dataset
print('generating test data')
x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

# set classifiers to 'svm' to test SVM classifier
# set classifiers to 'softmax' to test softmax classifier
# set classifiers to 'knn' to test kNN classifier
classifiers = 'svm'

if classifiers == 'svm':
    print('training SVM classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(svm_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))
    print('testing SVM classifier...')

    Wb = result.x
    print('accuracy of SVM loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

elif classifiers == 'softmax':
    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class,n_test)*100,'%')

else:  # knn
    # k value for kNN classifier. k can be either 1 or 3.
    k = 3
    print('testing kNN classifier...')
    print('accuracy of kNN loss: ', knn_test(x_train, y_train, x_test, y_test, n_train, n_test, k)*100
          , '% for k value of ', k)
