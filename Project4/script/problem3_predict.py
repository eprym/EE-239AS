# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:07:37 2016

@author: YuLiqiang
"""

import math
from math import sqrt
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import statsmodels.api as sm
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

from pybrain import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.neuronlayer import *




# Compare the linear regression model and random forest regression model
def linear_Regression(data, target, network):
    lr = linear_model.LinearRegression(normalize = True)
    
#    lr.fit(data, target)
#    F, pval = f_regression(data, lr.predict(data))
#    lr_predict=lr.predict(data)
#    rmse_linear = sqrt(np.mean((lr_predict - target) ** 2))
#    RMSE_LINEAR.append(rmse_linear)    
#    rfr = RandomForestRegressor(n_estimators = 30,max_depth = 12, max_features='auto')
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        
#        rfr = rfr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)
#        rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
#        RMSE_RFR.append(rmse_rfr)
    lr_predict=lr.predict(data)
    #scores = cross_validation.cross_val_score(rfr,data_test, target_test.ravel, cv=10)
    #print np.mean(scores)
    
#    print(np.mean(RMSE_RFR))
#    test_times = np.arange(1,11)
#    plt.figure()
#    plt.plot(test_times, RMSE_LINEAR, label = "RMSE in linear regression with 10-fold cv")
#    plt.plot(test_times, RMSE_RFR, label = "RMSE in random forest regression with 10-fold cv")
#    plt.ylim(0.0, 0.12)
    #plt.title("RMSE comparison between linear regression and random forest regression")
#    plt.xlabel("cross validation times")
#    plt.ylabel("RMSE")
#    plt.legend()

#    network['predicted_lr'] = lr.predict(data);
#    network['predicted_rfr'] = rfr.predict(data);
#    network_time_target = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["Size of Backup (GB)"].sum()
#    network_time_predict_lr = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["predicted_lr"].sum() 
#    network_time_predict_rfr = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["predicted_rfr"].sum()

#    time = np.arange(1, len(network_time_target)+1)
#    plt.figure()
#    plt.scatter(time, network_time_target, s = 15, color = 'red', label = "Actual values over time")
#    plt.scatter(time, network_time_predict_lr, s = 15, color = 'green', label = "predicted values with linear model")
#    plt.xlabel('Time')
#    plt.ylabel('Size of backup(GB)')
#    plt.ylim(-2,12)
#    plt.legend()
#
#    plt.figure()
#    plt.plot(time[0:120], network_time_predict_rfr[0:120], label = "predicted values with random forest tree model")
#    plt.legend()
#
#    plt.figure()
#    plt.scatter(lr.predict(data), lr.predict(data) - target, label = "residual VS fitted values")
#    plt.xlabel("fitted values")
#    plt.ylabel("residual")
#    plt.legend()
#    plt.show() 
    return RMSE_LINEAR,lr_predict
  

# Tuning the parameter of randomforest model, SLOW !!!   
def randomforest_tuning(data, target, network):
    RMSE_RFR=[]
    rfr = RandomForestRegressor(n_estimators = 100, max_features = 20, max_depth = 100)
    rfr.fit(data, target)
    rfr_predict=rfr.predict(data)
    rmse_rfr = sqrt(np.mean((rfr_predict - target) ** 2))
    RMSE_RFR.append(rmse_rfr)
#    kf = KFold(len(target), 10, shuffle = True);
#    RMSE_BEST = 10
#    rfr_best = RandomForestRegressor(n_estimators = 30, max_features = len(data[0]), max_depth = 8)
#    for nEstimators in range(1,31,1):
#        for maxFeatures in range(6, len(data[0]+1)):
#            for maxDepth in range(1,13,1):
#                rfr = RandomForestRegressor(n_estimators = nEstimators, max_features = maxFeatures, max_depth = maxDepth)
#                RMSE_RFR = []
#                for train_index, test_index in kf:
#                    data_train, data_test = data[train_index], data[test_index]
#                    target_train, target_test = target[train_index], target[test_index]
#                    rfr.fit(data_train, target_train)
#                    rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
#                    RMSE_RFR.append(rmse_rfr)
#                if RMSE_BEST > np.mean(RMSE_RFR):
#                    rfr_best = rfr
#                    RMSE_BEST = np.mean(RMSE_RFR)
#    kf_final = KFold(len(target), 10, shuffle = True);
#    RMSE_FINAL = []
#    for train_index, test_index in kf_final:
#        data_train, data_test = data[train_index], data[test_index]
#        target_train, target_test = target[train_index], target[test_index]
#        rfr_best.fit(data_train, target_train)
#        rmse_rfr = sqrt(np.mean((rfr_best.predict(data_test) - target_test) ** 2))
#        RMSE_FINAL.append(rmse_rfr)
#    plt.figure()
#    plt.plot(range(1,len(RMSE_FINAL)+1), RMSE_FINAL)
#    plt.title("The best RMSE with random forest")
#    plt.xlabel("cross validation times")
#    plt.ylabel("RMSE")
#    plt.show()
#    print(np.mean(RMSE_FINAL))
    return RMSE_RFR

# Fit the data with neural network
def neural_network(data, target, network):
    DS = SupervisedDataSet(len(data[0]), 1)
    nn = buildNetwork(len(data[0]), 7, 1, bias = True)
    kf = KFold(len(target), 10, shuffle = True);
    RMSE_NN = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        for d,t in zip(data_train, target_train):
            DS.addSample(d, t)
        bpTrain = BackpropTrainer(nn,DS, verbose = True)
        #bpTrain.train()
        bpTrain.trainUntilConvergence(maxEpochs = 10)
        p = []
        for d_test in data_test:
            p.append(nn.activate(d_test))
        
        rmse_nn = sqrt(np.mean((p - target_test)**2))
        RMSE_NN.append(rmse_nn)
        DS.clear()
    time = range(1,11)
    plt.figure()
    plt.plot(time, RMSE_NN)
    plt.xlabel('cross-validation time')
    plt.ylabel('RMSE')
    plt.show()
    print(np.mean(RMSE_NN))

def neural_network_converg(data, target, network):
    DS = SupervisedDataSet(len(data[0]), 1)
    nn = buildNetwork(len(data[0]), 7, 1, bias = True, hiddenclass = SigmoidLayer, outclass = LinearLayer) 
    for d, t in zip(data, target):
         DS.addSample(d,t)
    Train, Test = DS.splitWithProportion(0.9)
    #data_train = Train['input']
    data_test = Test['input']
    #target_train = Train['target']
    target_test = Test['target']
    bpTrain = BackpropTrainer(nn,Train, verbose = True)
    #bpTrain.train()
    bpTrain.trainUntilConvergence(maxEpochs = 10)
    p = []
    for d_test in data_test:
        p.append(nn.activate(d_test))
        
    rmse_nn = sqrt(np.mean((p - target_test)**2)) 
    print(rmse_nn) 
    
def polynomial(data,target,network,deg):
    lr=linear_model.Lasso (alpha = 1)
##    lr = linear_model.LinearRegression(normalize = True)    
    poly = PolynomialFeatures(degree=deg)
    data_poly=poly.fit_transform(data)
#    lr.fit(data_poly,target)
#    poly_predict=lr.predict(data_poly);
#    RMSE_poly=sqrt(np.mean((lr.predict(data_poly) - target) ** 2))
#    F, pval = f_regression(data_poly, lr.predict(data_poly))
    ### cross validation
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_POLY = []
    for train_index, test_index in kf:
        data_train, data_test = data_poly[train_index], data_poly[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        poly_predict=lr.predict(data_poly);
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_POLY.append(rmse_linear)
        
    RMSE_poly=np.mean(RMSE_POLY)
    return RMSE_POLY,poly_predict
    
def main():
    hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
    tag_start=5
    tag_num=1
    for i in range(tag_start,tag_start+tag_num):
        filename = 'problem3_data_#%s' %hashtag[i]
        data_tmp = np.loadtxt(filename)
        if(i == tag_start):
            network = data_tmp
        else:
            network = np.concatenate((network,data_tmp))
    
    data = network[0:len(network)-1,1:]
    
    target =network[1:,0]
    data=np.nan_to_num(data)
    data = StandardScaler().fit_transform(data)
    
#    plt.figure()
#    plt.scatter(np.linspace(0,len(target)-1,len(target)), target, label = "# tweet VS time")
#    plt.xlabel("time")
#    plt.ylabel("# tweet")
#    plt.legend()
#    plt.show()
    
#    print(np.isnan(np.sum(data)))
#    print(np.isnan(np.sum(target)))
    
    ### linear regression using statsmodel
#    model = sm.OLS(target, data)
#    results = model.fit()
#    print(results.summary())
    ### if you want to make the data into binary format, uncomment the line below
    #data = preprocessed_1(network)
    
    ### compare the linear model with the random forest model
    RMSE_linear,lr_predict=linear_Regression(data, target, network)
    print(np.mean(RMSE_linear))
    
    ### polynomial model
    RMSE_poly,ploy_predict=polynomial(data,target,network,2)
    print(np.mean(RMSE_poly))    
    
    ### Tuning the parameters of the random forest, SLOW !!!
#    RMSE_rfr=randomforest_tuning(data, target, network)
#    print(RMSE_rfr)
    
    
    ### use the neural network model to fit the data
#    neural_network(data, target, network)

if __name__ == "__main__":
    main()