# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:05:21 2016

@author: fengjun
"""

import json
import datetime, time
import math
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model

#files={'sample1_period1.txt':1,'sample2_period2.txt':2,'sample3_period3.txt':3,'sample4_period1.txt':1,'sample5_period1.txt':1,\
#        'sample6_period2.txt':2,'sample7_period3.txt':3,'sample1_period8.txt':1,'sample9_period2.txt':2,'sample10_period3.txt':3}
files={'sample2_period2.txt':2}
day_start = datetime.datetime(2015,01,16, 0,0,0)
day_start_time = int(time.mktime(day_start.timetuple()))

for (k,v) in  files.items():
    f = open('test_data/'+k)
    line = f.readline()
    tweet = json.loads(line)
    tweets = []
    while len(line)!=0:
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()
    end_date = datetime.datetime(2015,1,18, 0,0,0)
    start_date = datetime.datetime(2015,2,7, 23,59,59)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    num_tweets = len(tweets)
        
    for i in range(0, num_tweets):
        tweet = tweets[i]
        tweet_time = tweet['firstpost_date']
#        print("tweet_time is  {0:d}".format(tweet_time))
        if tweet_time < mintime:
            mintime=tweet_time
#            print("mintime is  {0:d}".format(mintime))
        if tweet_time > maxtime:
            maxtime=tweet_time
#            print("maxtime is  {0:d}".format(maxtime))
    print("file is {0:s}".format(k))
    print("mintime is  {0:d}".format(mintime))
    print("maxtime is  {0:d}".format(maxtime))
    print("hour is  {0:d}".format((mintime-maxtime)/3600))
    data = np.zeros(shape=(6,41))
    f = open('test_data/'+k)
    line = f.readline()
    
    while(len(line) != 0):
        tweet = json.loads(line)
        data_update = data[(tweet['firstpost_date']-mintime)/3600,:]
        data_update[0] += 1 # number of tweets
        data_update[1] += tweet['metrics']['citations']['total']  # number of retweets
        data_update[2] += tweet['author']['followers']  # number of followers
        data_update[3] = max(data_update[3], tweet['author']['followers'])  # max number of followers
        data_update[4] += len(tweet['tweet']['entities']['user_mentions'])  # user mention absolute number
        data_update[6] += len(tweet['tweet']['entities']['urls'])  # url number
        data_update[8] += tweet['original_author']['followers']  # total number of followers of the original authors
        data_update[9] = max(data_update[9], tweet['original_author']['followers'])  # max number of followers of the original authors
        data_update[10] += tweet['tweet']['favorite_count'] # total number of favorite
        
        # create time series for 20 minutes time period
        tweet_time = tweet['firstpost_date']
        if((tweet_time-mintime)/3600+2 < len(data)):
            data[(tweet_time-mintime)/3600+2][11+(tweet_time-day_start_time)%3600/1200] += 1
        if((tweet_time-mintime)/3600+1 < len(data)):   
            data[(tweet_time-mintime)/3600+1][14+(tweet_time-day_start_time)%3600/1200] += 1
        data_update[17+(tweet['firstpost_date']-day_start_time)/3600%24] = 1
        
        data[(tweet['firstpost_date']-mintime)/3600,:]=data_update
        line = f.readline()
    
    for j in range(len(data)):
        data[j][5] = (data[j][4]+0.0) / data[j][0]  # user mention per tweet
        data[j][7]= (data[j][6]+0.0) / data[j][0]   # url number per tweet
    
    feature = data
    target =data[1:,0]
    feature=np.nan_to_num(feature)
    feature = StandardScaler().fit_transform(feature)
    
    train_feature=feature[0:len(feature)-1,:]
    lr = linear_model.LinearRegression(normalize = True)
    lr.fit(train_feature, target)
    test_feature=feature[-1,:]
    lr_predict=lr.predict(test_feature)
    
    
    