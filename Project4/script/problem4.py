# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 07:45:13 2016

@author: fengjun
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
import datetime, time

def linear_Regression(data, target, network):
    lr = linear_model.LinearRegression(normalize = True)
#    lr=
    lr.fit(data, target)
#    F, pval = f_regression(data, lr.predict(data))
    lr_predict=lr.predict(data)
#    rmse_linear = sqrt(np.mean((lr_predict - target) ** 2))
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    ABS_LINEAR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        
#        rfr = rfr.fit(data_train, target_train)
        abs_linear = np.mean(abs(lr.predict(data_test) - target_test))
        ABS_LINEAR.append(abs_linear)
    print("average prediction error of cross validation is {0:f}".format(np.mean(ABS_LINEAR)))
    return ABS_LINEAR,lr_predict


def main():
    hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
    tag_start=1
    tag_num=1
    for i in range(tag_start,tag_start+tag_num):
        filename = 'problem2_data_#%s' %hashtag[i]
        data_tmp = np.loadtxt(filename)
        if(i == tag_start):
            network = data_tmp
        else:
            network = np.concatenate((network,data_tmp))
    
    data = network[0:len(network)-1,:]
    target =network[1:,0]
    data=np.nan_to_num(data)
    data = StandardScaler().fit_transform(data)
    linear_Regression(data, target, network)
    start_date = datetime.datetime(2015,01,16, 12,0,0)
    end_date = datetime.datetime(2015,02,07, 03,0,0)
    superbowl_start=datetime.datetime(2015,02,01, 8,0,0)
    superbowl_end=datetime.datetime(2015,02,01, 20,0,0)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    starttime = int(time.mktime(superbowl_start.timetuple()))
    endtime = int(time.mktime(superbowl_end.timetuple()))
    
    index1=(starttime-mintime)/3600
    index2=(endtime-mintime)/3600
    
    
    data = network[0:index1,:]
    target =network[1:index1+1,0]
    data=np.nan_to_num(data)
    data = StandardScaler().fit_transform(data)
    linear_Regression(data, target, network)
    
    data = network[index1:index2,:]
    target =network[index1+1:index2+1,0]
    data=np.nan_to_num(data)
    data = StandardScaler().fit_transform(data)
    linear_Regression(data, target, network)
    
    
    data = network[index2+1:-1,:]
    target =network[index2+2:,0]
    data=np.nan_to_num(data)
    data = StandardScaler().fit_transform(data)
    linear_Regression(data, target, network)
    
    
    
    

if __name__ == "__main__":
    main()
        