# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 20:51:02 2017

@author: Kara

This code is used for K-nearest Neighbors
"""

# import needed libraries, especially "minst"
import numpy as np 
from mnist import MNIST
# require `pip install python-mnist`
# https://pypi.python.org/pypi/python-mnist/
from sklearn import neighbors
from sklearn.metrics import accuracy_score
import time


# load the handwritten digit images for training and testing
# http://yann.lecun.com/exdb/mnist/
mndata = MNIST('MNIST/')
mndata.load_training()
mndata.load_testing()
X_test = mndata.test_images
X_train = mndata.train_images
y_test = np.asarray(mndata.test_labels)
y_train = np.asarray(mndata.train_labels)

# This is used to cluster number of classes in target
# There will be 10 classes for number from 0 to 9
print('Number of classes: %d' %len(np.unique(y_train)))
print('Training size: %d' %len(y_train))
print('Testing size: %d' %len(X_test))


# Starting with the case of learning from 1 nearest neighbor
start_time = time.time()
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
end_time = time.time()
print("Accuracy of 1NN for MNIST: %.2f %%" %(100*accuracy_score(y_test, y_pred)))
print("Running time: %.2f (s)" % (end_time - start_time))