# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 16:50:30 2016

@author: YuLiqiang
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups as f20
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware',
                'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']

train = f20(subset='train', shuffle = True, random_state = 42)
target = train.target
bins = np.linspace(-0.5, len(categories)-0.5, len(categories)+1)
bins_total = np.linspace(-0.5, 19.5, 21)
plt.hist(target, bins_total, rwidth = 0.7)
plt.show()

stop_words = text.ENGLISH_STOP_WORDS
count_vect = CountVectorizer(stop_words = stop_words)
train_vect = count_vect.fit_transform(train.data)
tfidf_transform = TfidfTransformer()
tfidf_vect = tfidf_transform.fit_transform(train_vect)
print(tfidf_vect.shape)


for cat in train.target_names:
    print len(train[cat].data)
