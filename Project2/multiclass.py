# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:50:30 2016

@author: YKaimingWang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD as TSVD
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve, precision_score, recall_score
from sklearn.metrics import confusion_matrix
import nltk.stem

from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

from sklearn.linear_model import LogisticRegression

from sklearn.multiclass import OneVsOneClassifier as OVOC
from sklearn.multiclass import OneVsRestClassifier as OVRC

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedTfidfVectorizer(TfidfVectorizer):

    def build_analyzer(self):
        analyzer = super(StemmedTfidfVectorizer, self).build_analyzer()
        return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

cat = ['comp.sys.ibm.pc.hardware','comp.sys.mac.hardware',
       'misc.forsale','soc.religion.christian']

train = f20(subset='train',categories=cat,shuffle =True, random_state = 42)
#train_comp = f20(subset='train',categories=cat[0:4], shuffle = True, random_state = 42)
#train_rect = f20(subset='train',categories=cat[4:], shuffle = True, random_state = 42)
#data=train.data
#target = train.target
#bins = np.linspace(-0.5, len(categories)-0.5, len(categories)+1)
#bins_total = np.linspace(-0.5, 19.5, 21)
#plt.hist(target, bins_total, rwidth = 0.7)
#plt.show()

stopwords = text.ENGLISH_STOP_WORDS
vectorizer = StemmedTfidfVectorizer(
    min_df=1, stop_words=stopwords, decode_error='ignore')
vector_train = vectorizer.fit_transform(train.data)
tfidf_train=vector_train.toarray()
svd = TSVD(n_components = 50, n_iter = 10, random_state = 42)  
tfidf_train_reduced = svd.fit_transform(tfidf_train)
#count_vect = CountVectorizer(stop_words = stop_words)
#train_vect = count_vect.fit_transform(train.data)
#tfidf_transform = TfidfTransformer()
#tfidf_vect = tfidf_transform.fit_transform(train_vect)
#print(tfidf_vect.shape)

#svd = TSVD(n_components = 50, n_iter = 10, random_state = 42)  
#tfidf_vect_reduced = svd.fit_transform(tfidf_vect)
#print(tfidf_vect_reduced.shape) 

svm_train_data = tfidf_train_reduced
#svm_train_tag = np.concatenate((-np.ones(len(train_comp.data)), np.ones(len(train_rect.data))))
svm_train_tag = train.target
#for i in train.target:
#    if(i < 4):  
#        svm_train_tag.append(-1)
#    else:
#        svm_train_tag.append(1)    

test = f20(subset='test', categories=cat, shuffle = True, random_state = 42)
#test_comp = f20(subset='test',categories=cat[0:4], shuffle = True, random_state = 42)
#test_rect = f20(subset='test',categories=cat[4:], shuffle = True, random_state = 42)
vector_test = vectorizer.fit_transform(test.data)
vector_test = vectorizer.fit_transform(test.data)
tfidf_test=vector_test.toarray() 
tfidf_test_reduced = svd.fit_transform(tfidf_test)
svm_test_data = tfidf_test_reduced
svm_test_tag = test.target
#for i in test.target:
#    if(i < 4):  
#        svm_test_tag.append(-1)
#    else:
#        svm_test_tag.append(1)
        
svc = SVC(kernel='linear',C = 100)
svc_ovoc=OVOC(svc)
svc_ovoc.fit(svm_train_data, svm_train_tag)
svc_ovoc_predict=svc_ovoc.predict(svm_test_data)
#precision, recall, thresholds = precision_recall_curve(svm_test_tag, svc_ovoc_predict)
#BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)
score=svc_ovoc.score(svm_test_data,svm_test_tag)
precision = precision_score(svm_test_tag, svc_ovoc_predict, average = 'weighted')
recall = recall_score(svm_test_tag, svc_ovoc_predict, average = 'weighted')
print "1 VS 1 SVC"
print "confusion matrix:","\n",confusion_matrix(svm_test_tag, svc_ovoc_predict)
print "score=",score
print "precision=", precision
print "recall=", recall
print '\n'

svc = SVC(kernel='rbf',C = 100)
svc_ovrc=OVRC(svc)
svc_ovrc.fit(svm_train_data, svm_train_tag)
svc_ovrc_predict=svc_ovrc.predict(svm_test_data)
#precision, recall, thresholds = precision_recall_curve(svm_test_tag, svc_ovoc_predict)
#BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)
score=svc_ovrc.score(svm_test_data,svm_test_tag)
precision = precision_score(svm_test_tag, svc_ovrc_predict, average = 'weighted')
recall = recall_score(svm_test_tag, svc_ovrc_predict, average = 'weighted')

print "1 VS Rest SVC"
print "confusion matrix:","\n",confusion_matrix(svm_test_tag, svc_ovrc_predict)
print "score=",score
print "precision=", precision
print "recall=", recall
print '\n'

#
gnb = GaussianNB()
gnb.fit(svm_train_data, svm_train_tag)
gnb_predict=gnb.predict(svm_test_data)
score = gnb.score(svm_test_data, svm_test_tag)
precision = precision_score(svm_test_tag, gnb_predict, average = 'weighted')
recall = recall_score(svm_test_tag, gnb_predict, average = 'weighted')
print "GaussianNB"
print "confusion matrix:","\n",confusion_matrix(svm_test_tag, gnb_predict)
print "score=",score
print "precision=", precision
print "recall=", recall
print '\n'

#mnb = MultinomialNB()
#mnb.fit(tfidf_train, svm_train_tag)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)
#score = mnb.score(tfidf_test, svm_test_tag)
#print score
#

bnb = BernoulliNB(alpha=1.0, binarize=0.0, class_prior=None, fit_prior=True)
bnb.fit(svm_train_data, svm_train_tag)
bnb_predict=bnb.predict(svm_test_data)
#precision, recall, thresholds = precision_recall_curve(svm_test_tag, ovoc_predict)
#BernoulliNB(alpha=1.0, binarize=0.5, class_prior=None, fit_prior=True)
score=bnb.score(svm_test_data,svm_test_tag)
precision = precision_score(svm_test_tag, bnb_predict, average = 'weighted')
recall = recall_score(svm_test_tag, bnb_predict, average = 'weighted')
print "BernoulliNB"
print "confusion matrix:","\n",confusion_matrix(svm_test_tag, bnb_predict)
print "score=",score
print "precision=", precision
print "recall=", recall
print '\n'
#print "precision=",precision[1]
#print "recall=",recall[1]
