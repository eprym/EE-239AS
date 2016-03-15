# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 23:24:53 2016

@author: YuLiqiang
"""

import json
import numpy as np
import math
import datetime, time

hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
data = np.zeros(shape=(519,41))
start_date = datetime.datetime(2015,01,16, 12,0,0)
end_date = datetime.datetime(2015,02,07, 03,0,0)
day_start = datetime.datetime(2015,01,16, 0,0,0)
mintime = int(time.mktime(start_date.timetuple()))
maxtime = int(time.mktime(end_date.timetuple()))
day_start_time = int(time.mktime(day_start.timetuple()))

for i in range(len(hashtag)):
    filename = '../tweet_data/tweets_#%s.txt' % hashtag[i]
    f = open(filename)
    line = f.readline()
    while(len(line) != 0):
        tweet = json.loads(line)
        if(tweet['firstpost_date'] < mintime or tweet['firstpost_date'] > maxtime):
            line = f.readline()
            continue
        data_update = data[(tweet['firstpost_date']-mintime)/3600]
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
        line = f.readline()
    
    for j in range(len(data)):
        data[j][5] = (data[j][4]+0.0) / data[j][0]  # user mention per tweet
        data[j][7]= (data[j][6]+0.0) / data[j][0]   # url number per tweet
    
    np.savetxt('problem3_data_#%s' %hashtag[i], data)
    print 'save the data succesfully for #%s\n' %hashtag[i]