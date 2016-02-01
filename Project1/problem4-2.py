# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 09:24:16 2016

@author: fengjun
"""
from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
from math import log10
from sklearn.cross_validation import KFold
from sklearn import linear_model
#from sklearn.cross_validation import cross_val_predict
#import the csv file
boston = datasets.load_boston()



#linear-regression
lr = linear_model.LinearRegression()

test_times = range(0,10)

plt.figure(1)
#divide the data into two sets
Train_set_x=pd.DataFrame(boston.data,index=None)
Train_set_y=pd.DataFrame(boston.target,index=None)
#split the data into 10 folds
f10 = KFold(len(Train_set_x), n_folds=10, shuffle=True, random_state=None)
# result1 linear,result2 poly-2,result3 poly-3,result4 poly-4,result5 poly-5,result6 poly-6,result7 poly-7  

for i in range(1,8):
    results = []
    for train_index, test_index in f10:
        x_train, x_test = Train_set_x.iloc[train_index], Train_set_x.iloc[test_index]
        y_train, y_test = Train_set_y.iloc[train_index], Train_set_y.iloc[test_index]

        poly = PolynomialFeatures(degree=i)
        x_train_2=poly.fit_transform(x_train)
        x_test_2=poly.fit_transform(x_test)
        lr.fit(x_train_2,y_train)
        error=log10(sqrt(np.mean((lr.predict(x_test_2) - y_test) ** 2)))
        results.append(error)
    print results
    plt.plot(test_times,results,label=('RMSE of Ploynomial Regression--Degree'+str(i)))
plt.legend(fontsize=6)
plt.title('Comparsion of Linear Regression and Polynomial Regression--Boston')
plt.xlabel('Test Times')
plt.ylabel('RMSE')
plt.ylim(0.0,7)
plt.show()
plt.savefig('problem4-2') 


#predicted = cross_val_predict(lr, Train_set_x, Train_set_y, cv=10)
#
#fig, ax = plt.subplots()
#ax.scatter(Train_set_y, predicted)
#ax.plot([Train_set_y.min(), Train_set_y.max()], [Train_set_y.min(), Train_set_y.max()], 'k--', lw=4)
#ax.set_xlabel('Measured')
#ax.set_ylabel('Predicted')
#plt.show()
