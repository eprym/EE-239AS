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
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem
import scipy

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        
class StemmedCountVectorizer(CountVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        
train = f20(subset='train',shuffle = True, random_state = 42)
datalist=[]
for i in range(0,20):
    datalist.append('')
for i in range(0,len(train.data)):
    datalist[train.target[i]]+=(' '+train.data[i])
    

stopwords = text.ENGLISH_STOP_WORDS
vectorizer = StemmedCountVectorizer(
    min_df=1, stop_words=stopwords, decode_error='ignore')
vector = vectorizer.fit_transform(datalist)
count=vector.toarray()

index={0:3,1:4,2:6,3:15}
tf=np.ndarray([4, len(count[0])], dtype=float)
icf=np.ndarray([4, len(count[0])], dtype=float)
for i in index:
    s=float(np.sum(count[index[i]]))
    for j in range(0,len(count[0])):
        tf[i][j]=0.5+0.5*count[index[i]][j]/s
for i in index:
    for j in range(0,len(count[0])):
        c=0
        for x in range(0,20):
            if(count[x][j]!=0):
                c=1+c
        icf[i][j]=scipy.log(float(20)/c)
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