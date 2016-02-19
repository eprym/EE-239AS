# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:03:45 2016

@author: Kami
"""
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt

from math import sqrt

import sklearn
from sklearn.preprocessing import normalize
from sklearn import cross_validation
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.cross_validation import KFold

from pybrain import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.neuronlayer import *


from preprocess2 import preprocess_2

def linear_regression(data, target):
    lr = linear_model.LinearRegression(normalize = True)
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)
    
    #scores = cross_validation.cross_val_score(rfr,data_test, target_test.ravel, cv=10)
    #print np.mean(scores)
    
    F, pval = f_regression(data_test, lr.predict(data_test))
    print(pval)
    
    test_times = np.arange(1,11)
    plt.figure()
    plt.plot(test_times, RMSE_LINEAR, label = "RMSE in linear regression with 10-fold cv")
#    plt.ylim(0.0, 0.12)
    #plt.title("RMSE comparison between linear regression and random forest regression")
    plt.xlabel("cross validation times")
    plt.ylabel("RMSE")
    plt.legend()

    predicted = lr.predict(data);
    index = np.arange(1, len(predicted)+1)

    plt.figure()
    plt.scatter(index, target, s = 15, color = 'red', label = "Actual")
    plt.scatter(index, predicted, s = 15, color = 'green', label = "Fitted")
    plt.xlabel('Index')
    plt.ylabel('MEDV')
    plt.legend()


    plt.figure()
    plt.scatter(predicted,predicted-target,label = "residual VS fitted values")
    plt.xlabel("fitted values")
    plt.ylabel("residual")
    plt.legend()
#    plt.ylim(-0.8,0.4)
    plt.show() 
    return RMSE_LINEAR
    

def main():
#    network=pa.read_csv("housing_data.csv",header=None)
    data = pa.read_csv("housing_data.csv",header=None).values[:,:]
    binaryData=preprocess_2(data)
    target=data[:,13]
    rmse_linear=linear_regression(binaryData,target)
    #rmse_l1CV=lassoCV_regression(binaryData,target)
    print("Mean RMSE of linear regression="+str(np.mean(rmse_linear)))
    #print(np.mean(rmse_l1CV))
    
if __name__ == "__main__":
    main()
