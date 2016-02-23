# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 14:18:52 2016

@author: YuLiqiang
"""

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from matplotlib.ticker import FormatStrFormatter
from sklearn.decomposition import TruncatedSVD as TSVD
import nltk.stem
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, precision_score, recall_score
from sklearn import cross_validation
from sklearn.cross_validation import KFold

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))
        
cat=['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
train = f20(subset='train',categories=cat, shuffle = True, random_state = 42)
##train = f20(subset='train',shuffle = True, random_state = 42)

stopwords = text.ENGLISH_STOP_WORDS
vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words=stopwords, decode_error='ignore')
vector_train = vectorizer.fit_transform(train.data)
tfidf_train=vector_train.toarray()
svd = TSVD(n_components = 50, n_iter = 10, random_state = 42)  
tfidf_train_reduced = svd.fit_transform(tfidf_train)
#print(tfidf_train.shape)
#print(tfidf_train_reduced.shape) 

svm_train_data = tfidf_train_reduced
#svm_train_tag = np.concatenate((-np.ones(len(train_comp.data)), np.ones(len(train_rect.data))))
svm_train_tag = []
for i in train.target:
    if(i < 4):  
        svm_train_tag.append(0)
    else:
        svm_train_tag.append(1)
svm_train_tag = np.array(svm_train_tag)     

  
test = f20(subset='test',categories=cat, shuffle = True, random_state = 42)
vector_test = vectorizer.transform(test.data)
tfidf_test=vector_test.toarray() 
tfidf_test_reduced = svd.transform(tfidf_test)
svm_test_data = tfidf_test_reduced

svm_test_tag = []
for i in test.target:
    if(i < 4):  
        svm_test_tag.append(0)
    else:
        svm_test_tag.append(1)
svm_test_tag = np.array(svm_test_tag)
C = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
bestC = 0
bestScore = 0

data = np.concatenate((svm_train_data, svm_test_data))
tag = np.concatenate((svm_train_tag, svm_test_tag))

for c in C:
    svm_classfier = SVC(C=c)
    kf = KFold(len(data), n_folds=5, shuffle=True, random_state=None)
    score = 0
    for train_index, test_index in kf:
        data_train = data[train_index]
        data_test = data[test_index]
        tag_train = tag[train_index]
        tag_test = tag[test_index]
        svm_classfier.fit(data_train, tag_train)
        score += svm_classfier.score(data_test, tag_test)
    if(score/5 > bestScore):
        bestScore = score/5
        bestC = c
print "the best c = ", bestC
svm_classfier = SVC(C = bestC)
svm_classfier.fit(data_train, tag_train)
score = svm_classfier.score(data_test, tag_test)
print "score = ", score

predict = svm_classfier.predict(data_test)
confusionMatrix = confusion_matrix(tag_test, predict)
print "The confusion matrix is\n"
print confusionMatrix

precision = precision_score(tag_test, predict)
recall = recall_score(tag_test, predict)

print "precision = ",precision
print "recall = ", recall
    