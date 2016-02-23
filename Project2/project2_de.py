# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:32:05 2016

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
svm_train_tag = []
for i in train.target:
    if(i < 4):  
        svm_train_tag.append(0)
    else:
        svm_train_tag.append(1)
svm_train_tag = np.array(svm_train_tag)


        
svm_classfier = SVC(C=100000)
svm_classfier.fit(svm_train_data, svm_train_tag)


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

predict = svm_classfier.predict(svm_test_data) 
confusionMatrix = confusion_matrix(svm_test_tag, predict)
print confusionMatrix   
score = svm_classfier.score(svm_test_data, svm_test_tag)
print "score = ", score

precision = precision_score(svm_test_tag, predict)
recall = recall_score(svm_test_tag, predict)
print "precision = ",precision
print "recall = ", recall



test_score = svm_classfier.decision_function(svm_test_data)
#roc_auc = auc(fpr, tpr)
fpr, tpr, thresholds = roc_curve(svm_test_tag, test_score)
plt.figure()
plt.plot(fpr, tpr, lw=3)
#plt.legend(loc = 'lower right')
plt.title("ROC curve")
plt.show()


