# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 19:34:39 2016

@author: fengjun
"""

import json
import datetime, time
import math
import numpy as np
import statsmodels.api as sm
from sklearn import linear_model
from sklearn.feature_selection import f_regression

from scipy import stats

def linear_Regression(data, target, network):
    lr = linear_model.LinearRegression(normalize = True)
#    lr=
    lr.fit(data, target)
#    F, pval = f_regression(data, lr.predict(data))
    lr_predict=lr.predict(data)
#    rmse_linear = sqrt(np.mean((lr_predict - target) ** 2))
    kf = KFold(len(target), n_folds=10, shuffle=True, random_state=None)
    RMSE_LINEAR = []
    for train_index, test_index in kf:
        data_train, data_test = data[train_index], data[test_index]
        target_train, target_test = target[train_index], target[test_index]
        lr.fit(data_train, target_train)
        
#        rfr = rfr.fit(data_train, target_train)
        rmse_linear = sqrt(np.mean((lr.predict(data_test) - target_test) ** 2))
        RMSE_LINEAR.append(rmse_linear)
    return RMSE_LINEAR,lr_predict

hashtag = ['gopatriots', 'gohawks', 'nfl', 'patriots', 'sb49', 'superbowl']
for j in range(1):
    f = open('../tweet_data/tweets_#'+hashtag[j]+'.txt')
    line = f.readline()
    tweet = json.loads(line)
    tweets = []
    while len(line)!=0:
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()
    
    start_date = datetime.datetime(2015,1,18, 15,30,0)
    end_date = datetime.datetime(2015,2,8, 15,30,00)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    data=np.zeros(shape=(504,28))
    num_tweets = len(tweets)
    for i in range(504):
#        data[i][4]=i%24
        data[i][4+i%24]=1
    for i in range(0, num_tweets):
        tweet = tweets[i]
        tweet_time = tweet['firstpost_date']
        if tweet_time >= mintime:
            if tweet_time >= maxtime:
                break;
        hour=math.floor((tweet['firstpost_date']-mintime)/3600)
        data[hour][0]+=1
        data[hour][1]+= tweet['metrics']['citations']['total']
        data[hour][2]+=tweet['author']['followers']
        data[hour][3] = max(data[hour][3], tweet['author']['followers'])
    X=data[0:-1,0:28]
    y=data[1:,0]
    mod = sm.OLS(y,X)
    res = mod.fit()
    print("****The Hashtag is {0:s}*******************".format(hashtag[j]))
    print res.summary()
    
    RMSE_LINEAR,lr_predict=linear_Regression(X,y,data)
    print(np.mean(RMSE_LINEAR))
    F, pval = f_regression(X, lr_predict)
    print(pval)
    print(stats.ttest_ind(y,y))
        
        
