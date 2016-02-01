# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:06:29 2016

@author: YuLiqiang
"""

from math import sqrt
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
import sklearn
from sklearn import linear_model
from sklearn.preprocessing import normalize
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression

from pybrain import *
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules.neuronlayer import *


# Binary the data that has no numerical meaning
def preprocessed_1(network):
    target_forgot=network.values[:,6:7]
    rowNO=len(network)
    ##transform the origin date into the proper forms
    ##the final result is in prepossessed
    ##meaning of its columns
    ##0:15 week #
    ##15:22 weekdays
    ##22:28 startTime
    ##28:33 workflow
    ##33:63 fileName
    ##63:64 Backup Time(hour)
    weeks=network.values[:,0:1]
    bweeks=np.zeros((rowNO,15))
    i=0
    for tmp in np.nditer(weeks):
        bweeks[i,int(tmp)-1]=1
        i+=1
    
    weekdays=network.values[:,1:2]
    bweekdays=np.zeros((rowNO,7))
    i=0
    for tmp in np.nditer(weekdays):
        bweekdays[i,int(tmp)-1]=1
        i+=1
        
    startTimes=network.values[:,2:3]
    bstartTimes=np.zeros((rowNO,6))
    i=0
    for tmp in np.nditer(startTimes):
        bstartTimes[i,int(tmp)/4]=1
        i+=1
        
    workflows=network.values[:,3:4]
    bworkflows=np.zeros((rowNO,5))
    i=0
    for tmp in np.nditer(workflows):
        bworkflows[i,int(tmp)]=1
        i+=1
        
    fileNames=network.values[:,4:5]
    bfileNames=np.zeros((rowNO,30))
    i=0
    for tmp in np.nditer(fileNames):
        bfileNames[i,int(tmp)]=1
        i+=1
        
    preprocessed=np.concatenate((bweeks,bweekdays,bstartTimes,bworkflows,bfileNames,target_forgot),axis=1)
    return preprocessed

# Compare the linear regression model and random forest regression model
def linearRegression(data, target, network):
    lr = linear_model.LinearRegression(normalize = True)
    rfr = RandomForestRegressor(n_estimators = 30,max_depth = 12, max_features='auto')
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    RMSE_RFR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rfr = rfr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)
        rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
        RMSE_RFR.append(rmse_rfr)
    
    #scores = cross_validation.cross_val_score(rfr,data_test, target_test.ravel, cv=10)
    #print np.mean(scores)
    
    F, pval = f_regression(data_test, lr.predict(data_test))
    print(np.mean(RMSE_RFR))
    test_times = np.arange(1,11)
    plt.figure()
    plt.plot(test_times, RMSE_LINEAR, label = "RMSE in linear regression with 10-fold cv")
    plt.plot(test_times, RMSE_RFR, label = "RMSE in random forest regression with 10-fold cv")
    plt.ylim(0.0, 0.12)
    #plt.title("RMSE comparison between linear regression and random forest regression")
    plt.xlabel("cross validation times")
    plt.ylabel("RMSE")
    plt.legend()

    network['predicted_lr'] = lr.predict(data);
    network['predicted_rfr'] = rfr.predict(data);
    network_time_target = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["Size of Backup (GB)"].sum()
    network_time_predict_lr = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["predicted_lr"].sum() 
    network_time_predict_rfr = network.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["predicted_rfr"].sum()
    time = np.arange(1, len(network_time_target)+1)

    plt.figure()
    plt.scatter(time, network_time_target, s = 15, color = 'red', label = "Actual values over time")
    plt.scatter(time, network_time_predict_lr, s = 15, color = 'green', label = "predicted values with linear model")
    plt.xlabel('Time')
    plt.ylabel('Size of backup(GB)')
    plt.legend()

    plt.figure()
    plt.plot(time, network_time_predict_rfr, label = "predicted values with random forest tree model")
    plt.legend()

    plt.figure()
    plt.scatter(lr.predict(data), lr.predict(data) - target, label = "residual VS fitted values")
    plt.xlabel("fitted values")
    plt.ylabel("residual")
    plt.legend()
    plt.ylim(-0.8,0.4)
    plt.show() 
    return RMSE_LINEAR
  

# Tuning the parameter of randomforest model, SLOW !!!   
def randomforest(data, target, network):
    kf = KFold(len(target), 10, shuffle = True);
    RMSE_BEST = 10
    rfr_best = RandomForestRegressor(n_estimators = 30, max_features = len(data[0]), max_depth = 8)
    for nEstimators in range(20,40,5):
        for maxFeatures in range(len(data[0])/2, len(data[0])):
            for maxDepth in range(4,15,2):
                rfr = RandomForestRegressor(n_estimators = nEstimators, max_features = maxFeatures, max_depth = maxDepth)
                RMSE_RFR = []
                for train_index, test_index in kf:
                    data_train, data_test = data[train_index], data[test_index]
                    target_train, target_test = target[train_index], target[test_index]
                    rfr.fit(data_train, target_train)
                    rmse_rfr = sqrt(np.mean((rfr.predict(data_test) - target_test) ** 2))
                    RMSE_RFR.append(rmse_rfr)
                if RMSE_BEST > np.mean(RMSE_RFR):
                    rfr_best = rfr
                    RMSE_BEST = np.mean(RMSE_RFR)
    kf_final = KFold(len(target), 10, shuffle = True);
    RMSE_FINAL = []
    for train_index, test_index in kf_final:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        rfr_best.fit(data_train, target_train)
        rmse_rfr = sqrt(np.mean((rfr_best.predict(data_test) - target_test) ** 2))
        RMSE_FINAL.append(rmse_rfr)
    return RMSE_FINAL

# Fit the data with neural network
def neural_network(data, target, network):
    DS = SupervisedDataSet(len(data[0]), 1)
    nn = buildNetwork(len(data[0]), 3, 1, bias = True)
    kf = KFold(len(target), 2, shuffle = True);
    RMSE_NN = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        for d,t in zip(data_train, target_train):
            DS.addSample(d, t)
        bpTrain = BackpropTrainer(nn,DS, verbose = True)
        #bpTrain.trainEpochs(epochs = 2)
        bpTrain.trainUntilConvergence(maxEpochs = 20)
        p = []
        for d_test in data_test:
            p.append(nn.activate(d_test))
        
        rmse_nn = sqrt(np.mean((p - target_test)**2))
        RMSE_NN.append(rmse_nn)
        DS.clear()
    print(np.mean(RMSE_NN))

def neural_network_converg(data, target, network):
    DS = SupervisedDataSet(len(data[0]), 1)
    nn = buildNetwork(len(data[0]), 3, 1, bias = True, hiddenclass = SigmoidLayer, outclass = LinearLayer) 
    for d, t in zip(data, target):
         DS.addSample(d,t)
    Train, Test = DS.splitWithProportion(0.9)
    #data_train = Train['input']
    data_test = Test['input']
    #target_train = Train['target']
    target_test = Test['target']
    bpTrain = BackpropTrainer(nn,Train, verbose = True)
    bpTrain.trainUntilConvergence(maxEpochs = 100)
    p = []
    for d_test in data_test:
        p.append(nn.activate(d_test))
        
    rmse_nn = sqrt(np.mean((p - target_test)**2)) 
    print(rmse_nn)       
def main():
    network = pa.read_csv("network_backup_dataset.csv", header = 0)
    dict = {"Monday" : "1", "Tuesday":"2", "Wednesday":"3", "Thursday":"4", "Friday":"5", "Saturday":"6", "Sunday":"7"}
    for i in dict:
        network['Day of Week']=[s.replace(i, dict[i]) for s in network['Day of Week']]
  
    network['File Name']=[s.replace("File_","") for s in network['File Name']]
    network['Work-Flow-ID']=[s.replace("work_flow_","") for s in network['Work-Flow-ID']]
    network['Day of Week']=[int(s)-1 for s in network['Day of Week']]
    network['File Name']=[int(s) for s in network['File Name']]
    network['Work-Flow-ID'] = [int(s) for s in network['Work-Flow-ID']]

    data1 = network.values[:, 0:5]
    data2 = network.values[:, 6:7]
    data = np.concatenate((data1, data2), axis=1)
    target = network.values[:,5]
    ### if you want to make the data into binary format, uncomment the line below
    data = preprocessed_1(network)
#    rmse = linearRegression(data, target, network)
#    print(np.mean(rmse))
#    RMSE_RFR = randomforest(data, target, network)
#    plt.figure()
#    plt.plot(range(1,len(RMSE_RFR)+1), RMSE_RFR)
#    plt.show()
#    print(np.mean(RMSE_RFR))
    neural_network_converg(data, target, network)
    


    
if __name__ == "__main__":
    main()
    
    