# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:24:48 2016

@author: YuLiqiang
"""

import json
import math
import datetime, time
from matplotlib import pyplot as plt

def loadTweet(txtname):
    filename = '../tweet_data/tweets_#%s.txt' % txtname
    f = open(filename)
    line = f.readline()
    tweet = json.loads(line)
    
    tweets = []
    while(len(line) != 0):
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()
    return tweets
        
        

hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
for i in range(1):
    tweets = loadTweet(hashtag[i])
    start_date = datetime.datetime(2015,02,01, 12,0,0)
    end_date = datetime.datetime(2015,02,01, 12,0,0)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    usrid = set()
    num_followers = 0
    num_retweet = 0
    for j in range(0, len(tweets)):
        tweet = tweets[j]
        tweet_time = tweet['firstpost_date']
        new_mintime = min(mintime, tweet_time)
        if(mintime-new_mintime < 3600*24*15):
            mintime = new_mintime
        maxtime = max(maxtime, tweet_time)
        usrid.add(tweet['tweet']['user']['id'])
        num_followers += tweet['tweet']['user']['followers_count']
        num_retweet += tweet['tweet']['retweet_count']
    hourlength = math.ceil((maxtime-mintime)/3600)
    print 'The average number of tweets per hour for #%s is %f\n' %(hashtag[i], len(tweets)/hourlength)
    print 'The average number of followers of users for #%s is %f\n' %(hashtag[i], num_followers/len(usrid))
    print 'The average number of retweet for #%s is %f\n' %(hashtag[i], (num_retweet+0.0)/len(tweets))

#hashtag_plot = ['nfl', 'superbowl']
#hashtag_plot = ['gohawks']
#for i in range(len(hashtag_plot)):
#    tweets = loadTweet(hashtag_plot[i])
#    start_date = datetime.datetime(2015,02,01, 12,0,0)
#    end_date = datetime.datetime(2015,02,01, 12,0,0)
#    mintime = int(time.mktime(start_date.timetuple()))
#    maxtime = int(time.mktime(end_date.timetuple())) 
#    
#    for j in range(0, len(tweets)):
#        tweet = tweets[j]
#        tweet_time = tweet['firstpost_date']
#        new_mintime = min(mintime, tweet_time)
#        if(mintime-new_mintime < 3600*24*15):
#            mintime = new_mintime
#        maxtime = max(maxtime, tweet_time)
#     
#    freq = []
#    for k in range(0, len(tweets)):
#        tweet = tweets[k]
#        if(tweet['firstpost_date'] < mintime):
#            continue
#        freq.append(math.ceil((tweet['firstpost_date']-mintime)/3600))
#    
#    plt.figure()    
#    plt.hist(freq, 494)
#    plt.xlabel('hours')
#    plt.ylabel('frequency')
#    plt.title('The number of tweets VS hour for #%s' %hashtag_plot[i])

