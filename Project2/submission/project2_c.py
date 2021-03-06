# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 15:39:27 2016

@author: fengjun
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 17:11:50 2016

@author: fengjun
"""
import numpy as np
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem.lancaster import LancasterStemmer
from nltk.tokenize.regexp import RegexpTokenizer
import scipy
class Tokenizer(object):  
    def __init__(self):
        self.tok=RegexpTokenizer(r'\b([a-zA-Z]+)\b')
        self.stemmer = LancasterStemmer()
    def __call__(self, doc):
        return [self.stemmer.stem(token) for token in self.tok.tokenize(doc)]        
# combine the documents of the same classes into the same docuemnt, divided by ' '        
train = f20(subset='train',shuffle = True, random_state = 42)
datalist=[]
for i in range(0,20):
    datalist.append('')
for i in range(0,len(train.data)):
    datalist[train.target[i]]+=(' '+train.data[i])
    
# get the count vector
stopwords = text.ENGLISH_STOP_WORDS
vectorizer = CountVectorizer(tokenizer = Tokenizer(),
                             stop_words=stopwords,
                             min_df=1)
vector = vectorizer.fit_transform(datalist)
count=vector.toarray()
# get the if and icf
index={0:3,1:4,2:6,3:15}
tf=np.ndarray([4, len(count[0])], dtype=float)
icf=np.ndarray([4, len(count[0])], dtype=float)
# calculating if
for i in index:
    s=float(np.sum(count[index[i]]))
    for j in range(0,len(count[0])):
        tf[i][j]=0.5+0.5*count[index[i]][j]/s
# calculating icf
for i in index:
    for j in range(0,len(count[0])):
        c=0
        for x in range(0,20):
            if(count[x][j]!=0):
                c=1+c
        icf[i][j]=scipy.log(float(20)/c)
# multiply if and icf, thus achiveing if.icf
# get the most 10 significent terms in the 4 classes
tficf=tf*icf
print '\n'+'comp.sys.ibm.pc.hardware'
indices = np.argsort(tficf[0])[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print top_features

print '\n'+'comp.sys.mac.hardware'
indices = np.argsort(tficf[1])[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print top_features

print '\n'+'misc.forsale'
indices = np.argsort(tficf[2])[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print top_features

print '\n'+'soc.religion.christian'
indices = np.argsort(tficf[3])[::-1]
features = vectorizer.get_feature_names()
top_n = 10
top_features = [features[i] for i in indices[:top_n]]
print top_features