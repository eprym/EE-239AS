# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 12:02:10 2016

@author: YuLiqiang
"""
import json
import math
import datetime, time
import numpy as np

hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
data = np.zeros(shape=(519,5))
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
        data_update[4] = (tweet['firstpost_date']-day_start_time)/3600%24 # time of date
        line = f.readline()

    np.savetxt('problem2_data_#%s' %hashtag[i], data)
    print 'save the data succesfully for #%s\n' %hashtag[i]
