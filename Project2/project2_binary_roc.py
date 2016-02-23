# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 21:44:50 2016

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
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression


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

svm_train_data = tfidf_train_reduced
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


svm_classfier = SVC(C=100000)
svm_classfier.fit(svm_train_data, svm_train_tag)
test_score = svm_classfier.decision_function(svm_test_data)
#roc_auc = auc(fpr, tpr)
fpr_hard, tpr_hard, thresholds_hard = roc_curve(svm_test_tag, test_score)

svm_classfier = SVC(C = 1000)
svm_classfier.fit(svm_train_data, svm_train_tag)
test_score = svm_classfier.decision_function(svm_test_data)
fpr_soft, tpr_soft, thresholds_soft = roc_curve(svm_test_tag, test_score)


bnb = BernoulliNB(alpha=1.0, binarize=-0.0075, class_prior=[0.5047619048,0.4952380952], fit_prior=True)
bnb.fit(svm_train_data, svm_train_tag)
bnb_predict=bnb.predict(svm_test_data)
test_score=bnb.predict_proba(svm_test_data)
bnb_fpr,bnb_tpr,bnb_thr=roc_curve(svm_test_tag,test_score[:,1])

gnb = GaussianNB()
gnb.fit(svm_train_data, svm_train_tag)
test_score=bnb.predict_proba(svm_test_data)
gnb_fpr,gnb_tpr,gnb_thr=roc_curve(svm_test_tag,test_score[:,1])


c1=1
lr1=LogisticRegression('l1',False,C=c1,warm_start=True,solver='liblinear')
lr1.fit(svm_train_data,svm_train_tag)
test_score=lr1.predict_proba(svm_test_data)
lr1_fpr,lr1_tpr,lr1_thr=roc_curve(svm_test_tag,test_score[:,1])
    
plt.figure()
plt.plot(fpr_hard, tpr_hard, lw=3, label = 'hard margin SVM')
plt.plot(fpr_soft, tpr_soft, lw=3, label = 'soft margin SVM')
plt.plot(bnb_fpr, bnb_tpr, lw=3, label = 'Bernoulli naive bayes classifier')
plt.plot(gnb_fpr, gnb_tpr, lw=3, label = 'Gaussian naive bayes classifier')
plt.plot(lr1_fpr, lr1_tpr, lw=3, label = 'logistic regression classifier')
plt.legend(loc = 'lower right')
plt.title("ROC curve")
plt.show()