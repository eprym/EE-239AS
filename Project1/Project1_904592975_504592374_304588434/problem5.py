# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 00:03:45 2016

@author: Kami
"""
import numpy as np
import pandas as pa
import matplotlib.pyplot as plt

from math import sqrt
from sklearn import linear_model
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.cross_validation import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

from preprocess2 import preprocess_2

def linear_regression(data, target):
    lr = linear_model.LinearRegression(normalize = True)
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    rmses = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rmse = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        rmses.append(rmse)
        
    
    return rmses

def ridge_regression(data,target,alphas):
    plt.figure()
    mean_rmses=[]
    kf=KFold(len(target),10,True,None)
    for alpha0 in alphas:
        rmses=[]
        clf=Ridge(alpha=alpha0,normalize=True,solver='svd')
        for train_index, test_index in kf:
            data_train,data_test=data[train_index],data[test_index]
            target_train,target_test=target[train_index],target[test_index]
            clf.fit(data_train,target_train)
            rmse=sqrt(np.mean((clf.predict(data_test)-target_test)**2))
            rmses.append(rmse)
            
        mean_rmses.append(np.mean(rmses))
        x0=np.arange(1,11)
        plt.plot(x0,rmses,label='alpha='+str(alpha0),marker='o')
        
    lr = linear_model.LinearRegression(normalize = True)
    rmses = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rmse = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        rmses.append(rmse)
    mean_rmses.append(np.mean(rmses))
    x0=np.arange(1,11)
    plt.plot(x0,rmses,label='linear',marker='*')
    
    plt.title("RMSE comparison between different alpha values of Ridge regularization")
    plt.legend()
    plt.show()
#    print(mean_rmses)
    return mean_rmses
        
def lasso_regression(data,target,alphas):
    plt.figure()
    mean_rmses=[]
    kf=KFold(len(target),10,True,None)
    for alpha0 in alphas:
        rmses=[]
        clf=Lasso(alpha=alpha0,normalize=True)
        for train_index, test_index in kf:
            data_train,data_test=data[train_index],data[test_index]
            target_train,target_test=target[train_index],target[test_index]
            clf.fit(data_train,target_train)
#            print(clf.sparse_coef_)
            rmse=sqrt(np.mean((clf.predict(data_test)-target_test)**2))
            rmses.append(rmse)
        mean_rmses.append(np.mean(rmses))
        x0=np.arange(1,11)
        plt.plot(x0,rmses,label='alpha='+str(alpha0),marker='o')
        
    lr = linear_model.LinearRegression(normalize = True)
    rmses = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        rmse = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        rmses.append(rmse)
    mean_rmses.append(np.mean(rmses))
    x0=np.arange(1,11)
    plt.plot(x0,rmses,label='linear',marker='*')
    
    plt.title("RMSE comparison between different alpha values of Lasso regularization")
    plt.xlabel("cross validation indices")
    plt.ylabel("RMSE")
    plt.legend()
    plt.show()
        
    return mean_rmses

def lassoCV_regression(data,target,alphas):
    clf=LassoCV()
    sfm = SelectFromModel(clf, threshold=0.25)
    sfm.fit(data, target)
    n_features = sfm.transform(data).shape[1]
    
    while n_features > 2:
        sfm.threshold += 0.1
        data_transform = sfm.transform(data)
        n_features = data_transform.shape[1]
     
    rmses=[]
    kf=KFold(len(target),10,True,None)
    for train_index, test_index in kf:
        data_train,data_test=data_transform[train_index],data_transform[test_index]
        target_train,target_test=target[train_index],target[test_index]
        clf.fit(data_train,target_train)
        rmse=sqrt(np.mean((clf.predict(data_test)-target_test)**2))
        rmses.append(rmse)
        
    x0=np.arange(1,11)
    
    plt.figure()
    plt.plot(x0,rmses,label='LassoCV')
    plt.legend()
    plt.show()
    
    return rmses
    

def main():
    data = pa.read_csv("housing_data.csv",header=None).values[:,:]
    binaryData=preprocess_2(data)
    target=data[:,13]
    alphas=np.array([0.1,0.01,0.001])
    rmse_linear=linear_regression(data[:,0:13],target,)
    rmse_l2=ridge_regression(binaryData,target,alphas)
    rmse_l1=lasso_regression(binaryData,target,alphas)
    #rmse_l1CV=lassoCV_regression(binaryData,target)
    print("Mean RMSE of linear regression="+str(np.mean(rmse_linear)))
    print("Mean RMSEs of Lasso regression="+str(rmse_l1))
    print("Mean RMSEs of Ridge regression="+str((rmse_l2)))
    #print(np.mean(rmse_l1CV))
    
if __name__ == "__main__":
    main()

#    dict = {"Monday" : "1", "Tuesday":"2", "Wednesday":"3", "Thursday":"4", "Friday":"5", "Saturday":"6", "Sunday":"7"}
#    for i in dict:
#        network['Day of Week']=[s.replace(i, dict[i]) for s in network['Day of Week']]
#      
#    network['File Name']=[s.replace("File_","") for s in network['File Name']]
#    network['Work-Flow-ID']=[s.replace("work_flow_","") for s in network['Work-Flow-ID']]
#    network['Day of Week']=[int(s) for s in network['Day of Week']]
#    network['File Name']=[int(s) for s in network['File Name']]
#    network['Work-Flow-ID'] = [int(s) for s in network['Work-Flow-ID']]