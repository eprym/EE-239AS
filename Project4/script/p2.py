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

hashtag = ['gopatriots', 'gohawks', 'nfl', 'patriots', 'sb49', 'superbowl']
for j in range(1):
    f = open('tweet_data/tweets_#'+hashtag[j]+'.txt')
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
    data=np.zeros(shape=(504,5))
    num_tweets = len(tweets)
    for i in range(504):
        data[i][4]=i%24
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
    X=data[0:-1]
    y=data[1:,0]
    mod = sm.OLS(y,X)
    res = mod.fit()
    print("****The Hashtag is {0:s}*******************".format(hashtag[j]))
    print res.summary()
        
        
