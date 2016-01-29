# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 14:53:22 2016

@author: YuLiqiang
"""

from sklearn import datasets
from sklearn.cross_validation import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt

lr = linear_model.LinearRegression()
boston = datasets.load_boston()
y = boston.target

# cross_val_predict returns an array of the same size as `y` where each entry
# is a prediction obtained by cross validated:
#predicted = cross_val_predict(lr, boston.data, y, cv=10)
lr.fit(boston.data, y)
print(lr.coef_)