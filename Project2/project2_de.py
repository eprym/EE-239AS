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
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix

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


        
svm_classfier = SVC(C=1000000)
svm_classfier.fit(svm_train_data, svm_train_tag)


test = f20(subset='test',categories=cat, shuffle = True, random_state = 42)
vector_test = vectorizer.fit_transform(test.data)
tfidf_test=vector_test.toarray() 
tfidf_test_reduced = svd.fit_transform(tfidf_test)
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
print score

min_precision = np.linspace(0.1, 0.9, 9)
test_score = svm_classfier.decision_function(svm_test_data)
precision, recall, thr = precision_recall_curve(svm_test_tag, test_score)
FPR = []
TPR = []
for minp in min_precision:
    min_thr = min([thr[i] for i in range(len(thr)) if precision[i] > minp])
    predict = [1 if test_s>min_thr else 0 for test_s in test_score]
    fpr, tpr, threshold = roc_curve(svm_test_tag, predict)
    FPR.append(fpr[1])
    TPR.append(tpr[1])
#roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(FPR, TPR, lw = 3)
#plt.legend(loc = 'lower right')
plt.title("ROC curve")
plt.show()
