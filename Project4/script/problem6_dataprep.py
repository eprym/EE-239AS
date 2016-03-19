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
import matplotlib.pyplot as plt


#hashtag = ['gohawks', 'patriots']
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
#        
#emoticons_str = r"""
#    (?:
#        [:=;] # Eyes
#        [oO\-]? # Nose (optional)
#        [D\)\]\(\]/\\OpP] # Mouth
#    )"""
# 
#regex_str = [
#    emoticons_str,
#    r'<[^>]+>', # HTML tags
#    r'(?:@[\w_]+)', # @-mentions
#    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
#    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
# 
#    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
#    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
#    r'(?:[\w_]+)', # other words
#    r'(?:\S)' # anything else
#]
#    
#tokens_re = re.compile(r'('+'|'.join(regex_str)+')', re.VERBOSE | re.IGNORECASE)
#emoticon_re = re.compile(r'^'+emoticons_str+'$', re.VERBOSE | re.IGNORECASE)
#
#def tokenize(s):
#    return tokens_re.findall(s)
# 
#def preprocess(s, lowercase=True):
#    tokens = tokenize(s)
#    if lowercase:
#        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
#    return tokens
#
#punctuation = list(string.punctuation)
#stop = stopwords.words('english') + punctuation + ['rt', 'via']
#
### building term list and term co-occurrence list for hawks
#count_list_hawks = []
#co_occur_list_hawks = []
#for tweets in tweets_hawks:
#    count_single_stop = Counter()
#    co_occur = defaultdict(lambda:defaultdict(int))
#    for tweet in tweets:
#        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
#        count_single_stop.update(terms_stop)
#        
#        for i in range(len(terms_stop)-1):
#            for j in range(i+1, len(terms_stop)):
#                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
#                if(w1 != w2):
#                    co_occur[w1][w2] += 1
#                    
#    #print(count_single_stop.most_common(10))
#    count_list_hawks.append(count_single_stop)
#    co_occur_list_hawks.append(co_occur)
#    
### building term list and term co-occurrence list for patriots    
#count_list_patriots = []
#co_occur_list_patriots = []
#for tweets in tweets_patriots:
#    count_single_stop = Counter()
#    co_occur = defaultdict(lambda:defaultdict(int))
#    for tweet in tweets:
#        terms_stop = [term for term in preprocess(tweet['tweet']['text']) if term not in stop]
#        count_single_stop.update(terms_stop)
#        
#        for i in range(len(terms_stop)-1):
#            for j in range(i+1, len(terms_stop)):
#                w1, w2 = sorted([terms_stop[i], terms_stop[j]])
#                if(w1 != w2):
#                    co_occur[w1][w2] += 1
#                    
#    #print(count_single_stop.most_common(10))
#    count_list_patriots.append(count_single_stop)
#    co_occur_list_patriots.append(co_occur)

    
# calculate the PMI for terms
pmi_hawks_list = []
pmi_patriots_list = []


positive_vocab = [
    'good', 'nice', 'great', 'awesome', 'outstanding',
    'fantastic', 'terrific', ':)', ':-)', 'like', 'love',':D', ':-D', 'victory'
    'win', 'wonderful', 'happy', 'champion','championship'
]
negative_vocab = [
    'bad', 'terrible', 'crap', 'useless', 'hate', ':(', ':-(','lost', 'defeat',
    'damn', 'fuck', 'suck', 'disappointing', 'lose', 'sorry', 'unhappy', 'sad'                                                       
]


for i in range(len(count_list_hawks)):
    p_t = {}
    p_t_com = defaultdict(lambda:defaultdict(int))
    length = float(len(tweets_hawks[i]))
    for term, n in count_list_hawks[i].items():
        p_t[term] = n/length
        for t2 in co_occur_list_hawks[i][term]:
            p_t_com[term][t2] = co_occur_list_hawks[i][term][t2]/length
            
    pmi = defaultdict(lambda: defaultdict(int))
    for t1 in p_t:
        for t2 in co_occur_list_hawks[i][t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log(p_t_com[t1][t2]/denom,2)
    semantic_orientation = {}
    for term, n in p_t.items():
        pos_asso = sum(pmi[term][tx] for tx in positive_vocab)
        neg_asso = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = pos_asso - neg_asso
    #pmi_hawks_list.append(sum(semantic_orientation[term] for term in ['#gohawks', '#hawks', 'hawks', '#seahawks', 'seahawks'] if term in semantic_orientation))
    pmi_hawks_list.append(sum(semantic_orientation[term] for term in semantic_orientation)/float(len(semantic_orientation)))
    #pmi_hawks_list.append(semantic_orientation['#gohawks'])
    
for i in range(len(count_list_patriots)):
    p_t = {}
    p_t_com = defaultdict(lambda:defaultdict(int))
    length = float(len(tweets_patriots[i]))
    for term, n in count_list_patriots[i].items():
        p_t[term] = n/length
        for t2 in co_occur_list_patriots[i][term]:
            p_t_com[term][t2] = co_occur_list_patriots[i][term][t2]/length
            
    pmi = defaultdict(lambda: defaultdict(int))
    for t1 in p_t:
        for t2 in co_occur_list_patriots[i][t1]:
            denom = p_t[t1] * p_t[t2]
            pmi[t1][t2] = math.log(p_t_com[t1][t2]/denom,2)
    semantic_orientation = {}
    for term, n in p_t.items():
        pos_asso = sum(pmi[term][tx] for tx in positive_vocab)
        neg_asso = sum(pmi[term][tx] for tx in negative_vocab)
        semantic_orientation[term] = pos_asso - neg_asso
    #pmi_patriots_list.append(sum(semantic_orientation[term] for term in ['#gopatriots', 'patriots', 'gopats', 'brady'] if term in semantic_orientation))
    pmi_patriots_list.append(sum(semantic_orientation[term] for term in semantic_orientation)/float(len(semantic_orientation)))
    #pmi_patriots_list.append(semantic_orientation['#patriots'])

plt.figure()
x = range(24)
plt.plot(x, pmi_hawks_list, lw = 3, label = "Hawks")
plt.plot(x, pmi_patriots_list, lw = 3, label = "Patriots")
plt.xlabel("Time(hour)")
plt.ylabel("Emotion")
plt.legend()
plt.title("Fans emotion during the day of the superbowl final")
plt.show()

plt.figure()
x = range(7)
plt.plot(x, pmi_hawks_list[14:21], lw = 3, label = "Hawks")
plt.plot(x, pmi_patriots_list[14:21], lw = 3, label = "Patriots")
plt.xlabel("Time(hour)")
plt.ylabel("Emotion")
plt.legend()
plt.title("Fans emotion within 7 hours during and after the game")
plt.show()