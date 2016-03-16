# -*- coding: utf-8 -*-
"""
Created on Sun Mar 13 22:53:43 2016

@author: fengjun
"""

import json
import datetime, time
import math
from matplotlib import pyplot as plt

hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
for j in range(len(hashtag)):
    f = open('tweet_data/tweets_#'+hashtag[j]+'.txt')
    line = f.readline()
    tweet = json.loads(line)
    tweets = []
    while len(line)!=0:
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()
    
    start_date = datetime.datetime(2015,1,18, 0,0,0)
    end_date = datetime.datetime(2015,2,7, 23,59,59)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    
    
    num_tweets = len(tweets)
    num_window = 0
    retweets=0
    
    followers = 0
    for i in range(0, num_tweets):
        tweet = tweets[i]
        tweet_time = tweet['firstpost_date']
        if tweet_time >= mintime:
            if tweet_time >= maxtime:
                break;
            num_window += 1
        followers =followers+tweet['author']['followers']
        retweets=retweets+ tweet['metrics']['citations']['total']
    
    print("****The Hashtag is {0:s}*******************".format(hashtag[j]))
    print("average number of tweets per hour from 2015:01:18 00:00:00 to 2015:02:07 23:59:59 is  {0:f}".format(float(num_window)/504))
    print("average number of followers of users posting the tweets is {0:f}".format(float(followers)/num_tweets))
    print("average number of retweets is {0:f}".format(float(retweets)/num_tweets))

hashtag=['nfl','superbowl']
for j in range(len(hashtag)):
    f = open('tweet_data/tweets_#'+hashtag[j]+'.txt')
    line = f.readline()
    tweet = json.loads(line)
    tweets = []
    while len(line)!=0:
        tweet = json.loads(line)
        tweets.append(tweet)
        line = f.readline()
    
    start_date = datetime.datetime(2015,1,18, 0,0,0)
    end_date = datetime.datetime(2015,2,7, 23,59,59)
    mintime = int(time.mktime(start_date.timetuple()))
    maxtime = int(time.mktime(end_date.timetuple()))
    
    
    num_tweets = len(tweets)
    freq = []
    followers = 0
    for i in range(0, num_tweets):
        tweet = tweets[i]
        tweet_time = tweet['firstpost_date']
        if tweet_time >= mintime:
            if tweet_time >= maxtime:
                break;
            freq.append(math.floor((tweet['firstpost_date']-mintime)/3600))
    plt.figure(j)    
    plt.hist(freq, 504)
    plt.xlabel('Hour #')
    plt.ylabel('Number of Tweets')
    plt.title('The Number of Tweets in Hours for Hashtag #%s' %hashtag[j])
