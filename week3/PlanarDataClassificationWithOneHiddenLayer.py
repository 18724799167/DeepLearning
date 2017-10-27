# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 22:51:36 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary,sigmoid,load_planar_dataset,load_extra_datasets

np.random.seed(1) #set a seed so that the results are consistent


def showaFlower():
    X,Y = load_planar_dataset()
    plt.scatter(X[0,:], X[1,:], c = Y,s = 40,cmap = plt.cm.Spectral)
    return X,Y

def testWithLR(X,Y):
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T,Y.T)
    plot_decision_boundary(lambda x: clf.predict(x),X,Y)
    plt.title("Logistic Regression")
    
    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) + ' % ' + "(percentage of correctly labelled datapoints)")

def layer_sizes(X,Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return (n_x,n_h,n_y)

def initialize_parameters(n_x,n_h,n_y):
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x) * 0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h) * 0.01
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h,1))
    assert (W2.shape == (n_y,n_h))
    assert (b2.shape == (n_y,1))
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2":b2}
    
    return parameters
def sigmoid(Z):
    
    s = 1 / (1 + np.exp(-Z))
    return s

def forward_propagation(X,parameters):
    
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    
    assert (A2.shape == (1, X.shape[1]))
    
    cache = {"Z1": Z1,'A1': A1,'Z2': Z2,'A2': A2}
    
    return A2, cache

def compute_cost(A2, Y, parameters):
    m = Y.shape[1]
    
    logprobs = np.multiply(np.log(A2),Y) +np.multiply(np.log(1 - A2),(1 - Y))
    cost = -np.sum(logprobs)
    
    cost = np.squeeze(cost) / m
    
    assert(isinstance(cost,float))

    return cost    
    
def backward_propagation(parameters,cache,X,Y):
    m = X.shape[1]
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    
    A1 = cache['A1']
    A2 = cache['A2']
    
    dz2 = A2 - Y
    dW2 = (1 / m) * np.dot(dz2,A1.T)
    db2 = (1 / m) * np.sum(dz2,axis = 1,keepdims = True)
    dz1 = np.dot(W2.T,dz2) * (1 - np.power(A1,2))
    dW1 = (1 / m) * np.dot(dz1,X.T)
    db1 = (1 / m) * np.sum(dz1,axis = 1,keepdims = True)
    
    grads = {'dW1':dW1,
              'db1':db1,
              'dW2':dW2,
              'db2':db2}
    
    return grads

def update_parameters(parameters,grads,learning_rate = 1.2):
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']
    
    W1 = W1 - learning_rate * dW1
    W2 = W2 - learning_rate * dW2
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    
    parameters = {"W1": W1,"b1": b1,"W2": W2,"b2": b2}
    
    return parameters

def nn_model(X,Y,n_h,num_iterations = 10000,print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]
    
    parameters = initialize_parameters(n_x,n_h,n_y)
    
    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']
    
    for i in range(0,num_iterations):
        A2,cache = forward_propagation(X,parameters)
        cost = compute_cost(A2,Y,parameters)
        grads = backward_propagation(parameters,cache,X,Y)
        parameters = update_parameters(parameters,grads,learning_rate= 1.2)
        
        if print_cost and i % 1000 == 0:
             print("Cost after iteration %i: %f" %(i,cost))
    
    return parameters

def predict(parameters,X):
    A2,cache = forward_propagation(X,parameters)
    predictions = (A2 > 0.5)
    
    return predictions

def test(X, Y, n_h = 4, num_iterations = 10000, print_cost=True):
    parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)
    
    plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
    
def testdifhid(X,Y,hidden_layer_sizes = [1,2,3,4,5,6,10,20], num_iterations = 10000, print_cost=True):
    #X,Y = load_planar_dataset()
    plt.figure(figsize = (16,32))
    
    for i,n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5,2,i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = nn_model(X,Y,n_h,num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters,x.T),X,Y)
        predictions = predict(parameters,X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1 - Y,1 - predictions.T)) / float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
    
def test_other_datasets():
    noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()
    datasets = {'noisy_circles':noisy_circles,'noisy_moons':noisy_moons,'blobs':blobs,'gaussian_quantiles':gaussian_quantiles}
    dataset = 'noisy_circles'
    
    X,Y = datasets[dataset]
    X,Y = X.T,Y.reshape(1,Y.shape[0])
    
    if dataset == 'blobs':
        Y = Y%2
    plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);  
    return X,Y