# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 21:44:38 2016

@author: YuLiqiang
"""

import json
import numpy as np
import math
import datetime, time
from nltk.tokenize import word_tokenize
import re
import operator
from collections import Counter, defaultdict
from nltk.corpus import stopwords
import string


#hashtag = ['gohawks', 'gopatriots']
#start_date = datetime.datetime(2015,02,01, 03,30,0)
#end_date = datetime.datetime(2015,02,02, 03,30,0)
#mintime = int(time.mktime(start_date.timetuple()))
#maxtime = int(time.mktime(end_date.timetuple()))
#
#tweets_hawks = [[] for i in range(24)]
#tweets_patriots = [[] for i in range(24)]
#
#for i in range(len(hashtag)):
#    filename = '../tweet_data/tweets_#%s.txt' % hashtag[i]
#    f = open(filename)
#    line = f.readline()
#    while(len(line) != 0):
#        tweet = json.loads(line)
#        if(tweet['firstpost_date'] < mintime or tweet['firstpost_date'] > maxtime):
#            line = f.readline()
#            continue
#        index = (tweet['firstpost_date']-mintime)/3600
#        if(hashtag[i] == 'gohawks'):
#            tweets_hawks[index].append(tweet)
#        else:
#            tweets_patriots[index].append(tweet)
#        line = f.readline()        
        
emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""
 
regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
 
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]
    
tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)

def tokenize(s):
    return tokens_re.findall(s)
 
def preprocess(s, lowercase=False):
    tokens = tokenize(s)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens

punctuation = list(string.punctuation)
stop = stopwords.words('english') + punctuation + ['rt', 'via']

## building term list and term co-occurrence list for hawks
count_list_hawks = []
co_occur_list_hawks = []
for tweets in tweets_hawks:
    count_single_stop = Counter()
    co_occur = defaultdict(lambda:defaultdict(int))
    for tweet in tweets:
        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
        count_single_stop.update(terms_stop)
        
        for i in range(len(terms_stop)-1):
            for j in range(i+1, len(terms_stop)):
                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
                if(w1 != w2):
                    co_occur[w1][w2] += 1
                    
    #print(count_single_stop.most_common(10))
    count_list_hawks.append(count_single_stop)
    co_occur_list_hawks.append(co_occur)
    
## building term list and term co-occurrence list for patriots    
count_list_patriots = []
co_occur_list_patriots = []
for tweets in tweets_patriots:
    count_single_stop = Counter()
    co_occur = defaultdict(lambda:defaultdict(int))
    for tweet in tweets:
        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
        count_single_stop.update(terms_stop)
        
        for i in range(len(terms_stop)-1):
            for j in range(i+1, len(terms_stop)):
                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
                if(w1 != w2):
                    co_occur[w1][w2] += 1
                    
    #print(count_single_stop.most_common(10))
    count_list_patriots.append(count_single_stop)
    co_occur_list_patriots.append(co_occur)