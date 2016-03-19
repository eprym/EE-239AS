# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 15:24:48 2016

@author: YuLiqiang
"""

import json
import math
import datetime, time
from matplotlib import pyplot as plt

#def loadTweet(txtname):
#    filename = '../tweet_data/tweets_#%s.txt' % txtname
#    f = open(filename)
#    line = f.readline()
#    tweet = json.loads(line)
#    
#    tweets = []
#    while(len(line) != 0):
#        tweet = json.loads(line)
#        tweets.append(tweet)
#        line = f.readline()
#    return tweets
        
        

hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
#for i in range(5,6):
#    tweets = loadTweet(hashtag[i])
#    start_date = datetime.datetime(2015,02,01, 12,0,0)
#    end_date = datetime.datetime(2015,02,01, 12,0,0)
#    mintime = int(time.mktime(start_date.timetuple()))
#    maxtime = int(time.mktime(end_date.timetuple()))
#    usrid = set()
#    num_followers = 0
#    num_retweet = 0
#    for j in range(0, len(tweets)):
#        tweet = tweets[j]
#        tweet_time = tweet['firstpost_date']
#        new_mintime = min(mintime, tweet_time)
#        if(mintime-new_mintime < 3600*24*15):
#            mintime = new_mintime
#        maxtime = max(maxtime, tweet_time)
#        usrid.add(tweet['tweet']['user']['id'])
#        num_followers += tweet['author']['followers']
#        num_retweet += tweet['metrics']['citations']['total']
#    hourlength = math.ceil((maxtime-mintime)/3600)
#    print 'The average number of tweets per hour for #%s is %f\n' %(hashtag[i], len(tweets)/hourlength)
#    print 'The average number of followers of users for #%s is %f\n' %(hashtag[i], num_followers/len(usrid))
#    print 'The average number of retweet for #%s is %f\n' %(hashtag[i], (num_retweet+0.0)/len(tweets))


#for i in range(0,len(hashtag)):
#    filename = '../tweet_data/tweets_#%s.txt' % hashtag[i]
#    f = open(filename)
#    line = f.readline()
#    
#    start_date = datetime.datetime(2015,02,01, 12,0,0)
#    end_date = datetime.datetime(2015,02,01, 12,0,0)
#    mintime = int(time.mktime(start_date.timetuple()))
#    maxtime = int(time.mktime(end_date.timetuple()))
#    usrid = set()
#    num_followers = 0
#    num_retweet = 0
#    num_tweet = 0;
#    
#    while(len(line) != 0):
#        tweet = json.loads(line)
#        num_tweet += 1
#        tweet_time = tweet['firstpost_date']
#        new_mintime = min(mintime, tweet_time)
#        if(mintime-new_mintime < 3600*24*15):
#            mintime = new_mintime
#        maxtime = max(maxtime, tweet_time)  
#        if(tweet['tweet']['user']['id_str'] not in usrid):
#            num_followers += tweet['author']['followers']
#            usrid.add(tweet['tweet']['user']['id_str'])
#        num_retweet += tweet['metrics']['citations']['total']
#        line = f.readline()
#    
#    f.close()
#    hourlength = math.ceil((maxtime-mintime)/3600)
#    print 'The average number of tweets per hour for #%s is %f\n' %(hashtag[i], num_tweet/float(hourlength))
#    print 'The average number of followers of users for #%s is %f\n' %(hashtag[i], num_followers/float(len(usrid)))
#    print 'The average number of retweet for #%s is %f\n' %(hashtag[i], (num_retweet+0.0)/num_tweet)
    
hashtag_plot = ['nfl', 'superbowl']
#hashtag_plot = ['gohawks']
for i in range(len(hashtag_plot)):
    filename = '../tweet_data/tweets_#%s.txt' % hashtag_plot[i]
    f = open(filename)
    line = f.readline()
    
    start_date = datetime.datetime(2015,02,01, 12,0,0)
    end_date = datetime.datetime(2015,02,01, 12,0,0)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple())) 
    
    while(len(line) != 0):
        tweet = json.loads(line)
        tweet_time = tweet['firstpost_date']
        new_mintime = min(mintime, tweet_time)
        if(mintime-new_mintime < 3600*24*15):
            mintime = new_mintime
        maxtime = max(maxtime, tweet_time)
        line = f.readline()
        
    f.close() 
    freq = []
    f = open(filename)
    line = f.readline()
    while(len(line) != 0):
        tweet = json.loads(line)
        if(tweet['firstpost_date'] < mintime):
            line = f.readline()
            continue
        freq.append(math.ceil((tweet['firstpost_date']-mintime)/3600))
        line = f.readline()
    f.close()
    
    plt.figure()    
    plt.hist(freq, 494)
    plt.xlabel('hours')
    plt.ylabel('frequency')
    plt.title('The number of tweets VS hour for #%s' %hashtag_plot[i])

