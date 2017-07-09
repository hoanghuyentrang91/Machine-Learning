# -*- coding: utf-8 -*-
"""
Created on Thu Apr 20 19:39:11 2017

@author: Kara

This code is used for K-nearest Neighbors
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn import neighbors, datasets


# Import Iris flower dataset from scikit-learn. 
# This dataset has 150 data points for 3 types of Iris flowers.
iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
# This is used to cluster number of classes in target
print('Number of classes: %d' %len(np.unique(iris_y)))
print('Number of data points: %d' %len(iris_y))


# Print some examples of each class
X0 = iris_X[iris_y == 0,:]
print('\nSamples from class 0:\n', X0[:5,:])

X1 = iris_X[iris_y == 1,:]
print('\nSamples from class 1:\n', X1[:5,:])

X2 = iris_X[iris_y == 2,:]
print('\nSamples from class 2:\n', X2[:5,:])


# Split the dataset into two parts: one for training, the other for testing
from sklearn.cross_validation import train_test_split
#from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)

print("Training size: %d" %len(y_train))
print("Test size    : %d" %len(y_test))


# Learn more about K-nearest Neighbor for suppervised learning at
# http://scikit-learn.org/stable/modules/classes.html#module-sklearn.neighbors
# http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier
# Try with number of neighbors is 1
clf = neighbors.KNeighborsClassifier(n_neighbors = 1, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 random test data points in case of learning from 1 neighbors:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])

# This code is used to evaluate the performance
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
from sklearn.metrics import accuracy_score
print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# Try again with number of neighbors is 10
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Print results for 20 random test data points in case of learning from 10 neighbors:")
print("Predicted labels: ", y_pred[20:40])
print("Ground truth    : ", y_test[20:40])

print("Accuracy of 10NN with major voting: %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# Try again with number of neighbors is 10. 
# However, in this time, try to have different weight for each neighbor.
# Closer neighbors of a query point will have a greater influence than neighbors which are further away.
clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = 'distance')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN (1/distance weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))


# Another way to have a weight corresponding to each neighbor is as follows:
def myweight(distances):
    sigma2 = .5 # we can change this number
    return np.exp(-distances**2/sigma2)

clf = neighbors.KNeighborsClassifier(n_neighbors = 10, p = 2, weights = myweight)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy of 10NN (customized weights): %.2f %%" %(100*accuracy_score(y_test, y_pred)))