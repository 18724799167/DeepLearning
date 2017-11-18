# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 10:18:52 2017

@author: Administrator
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

def show_picture(x,y,index,classes):
    plt.imshow(x[index])
    print('y = ' + str(x[0,index]) + ", it's a '" + classes[np.squeeze(y[0, index])].decode("utf-8") +  "' picture.")
    
def img2Vector(train_set_x_orig,train_set_y_orig,test_set_x_orig,test_set_y_orig,classes):
    m_train = train_set_x_orig.shape[0]
    m_test = test_set_x_orig.shape[0]
    num_px = train_set_x_orig.shape[1]
    #print('m_train: =',m_train)
    #print('m_test: =',m_test)
    #print('num_px: =',num_px)
    train_set_x_flatten = train_set_x_orig.reshape(m_train,-1).T
    test_set_x_flatten = test_set_x_orig.reshape(m_test,-1).T
    
    #print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
    #print ("train_set_y shape: " + str(train_set_y_orig.shape))
    #print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
    #print ("test_set_y shape: " + str(test_set_y_orig.shape))
    #print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    
    train_set_x = train_set_x_flatten/255
    test_set_x = test_set_x_flatten/255
    
    return train_set_x,train_set_y_orig,test_set_x,test_set_y_orig,num_px

def sigmoid(z):
    s = 1/(1 + np.exp(-z))
    return s

def initialize_with_zeros(dim):
    w = np.zeros([dim,1])
    b = 0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return w,b

def propagate(w,b,X,Y):
    
    m = X.shape[1]
    A = sigmoid(np.dot(w.T,X) + b)
    #print(A)
    #cost = ((-1 / m) * (np.dot(Y, np.log(A).T) + np.dot((1-Y), np.log(1-A).T)))
    cost = - 1/m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))        
        
    dw = (1/m) * (np.dot(X,(A - Y).T))
    db = ((1 / m) * (A - Y)).sum(axis = 1)

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {'dw':dw,'db':db}
    return grads,cost
    
def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = True):
    costs = []
    
    for i in range(num_iterations):
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i % 100 == 0:
            costs.append(cost)
        
        if print_cost and i % 100 == 0:
            print('Cost after iteration %i: %f' %(i,cost))
        
    params = {'w':w,'b':b}
    grads = {'dw':dw,'db':db}
    
    return params,grads,costs
    
def predict(w,b,X):
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X) + b)
    
    for i in range (A.shape[1]):
        Y_prediction[0,i] = 1 if A[0,i]>0.5 else 0
    assert(Y_prediction.shape == (1,m))
    return Y_prediction
        
def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate =0.005,print_cost = False):
    w,b = initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    w = parameters['w']
    b = parameters['b']
    
    Y_prediction_test = predict(w,b,X_test)
    Y_prediction_train = predict(w,b,X_train)
    
    print('train accuracy:{} %'.format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print('test accuracy:{} %'.format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))
    
    d = {'cost': costs,
         'Y_prediction_test':Y_prediction_test,
         'Y_prediction_train':Y_prediction_train,
         'w':w,
         'b':b,
         'learning_rate':learning_rate,
         'num_iterations':num_iterations}
    
    return d
    
def show_prediction(x,y,d,index,classes,train_set_x_orig):
    num_px = train_set_x_orig.shape[1]
    plt.imshow(x[:,index].reshape((num_px, num_px, 3)))
    print ("y = " + str(y[0,index]) + ", you predicted that it is a \"" + classes[int(d["Y_prediction_test"][:,index])].decode("utf-8") +  "\" picture.")  
    
def plot_costFunction_and_gradients(d):
    costs = np.squeeze(d['cost'])
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations(per hundreds)')
    plt.title('Learning rate =' + str(d['learning_rate']))
    plt.show
    
def predict_my(d,my_images,classes,num_px = 64):
    length = len(my_images)
    num = 1
    for img in my_images:
        fname = 'H:/deepLearning/LR/images/' + img
        img = np.array(ndimage.imread(fname,flatten = False))
        my_image = scipy.misc.imresize(img,size = (num_px,num_px)).reshape((1,num_px * num_px * 3)).T
        my_predicted_image = predict(d['w'],d['b'],my_image)
        plt.subplot(length,1,num)
        plt.imshow(img)
        num += 1
        print("y = " + str(np.squeeze(my_predicted_image)) + ", your algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")
    
    
    
    
    
    
    
    
    
    
    
    
    
    