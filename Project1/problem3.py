# -*- coding: utf-8 -*-
"""
Created on Sun Jan 31 09:24:16 2016

@author: fengjun
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from math import sqrt
from sklearn.cross_validation import KFold
from sklearn import linear_model
#import the csv file
df = pd.read_csv('network_backup_dataset.csv',header=0,index_col=None)


#preparing the data
dict = {"Monday" : "1", "Tuesday":"2", "Wednesday":"3", "Thursday":"4", "Friday":"5", "Saturday":"6", "Sunday":"7"}

for i in dict:
    df['Day of Week']=[s.replace(i, dict[i]) for s in df['Day of Week']]
df['Day of Week']=[int(s) for s in df['Day of Week']]
#
df['Work-Flow-ID']=[s.replace('work_flow_', '') for s in df['Work-Flow-ID']]
df['Work-Flow-ID']=[int(s) for s in df['Work-Flow-ID']]
df['File Name']=[s.replace('File_', '') for s in df['File Name']]
df['File Name']=[int(s) for s in df['File Name']]


#linear-regression
lr = linear_model.LinearRegression()

test_times = range(0,10)
plt.figure(1)

for i in range(0,5):
    df_p=df[df['Work-Flow-ID']==i]
    Train_set_x=df_p.drop('Size of Backup (GB)',1)
    Train_set_x=df_p.drop('Work-Flow-ID',1)
    Train_set_y=df_p.ix[:,'Size of Backup (GB)']
    results=[]
    f10 = KFold(len(Train_set_x), n_folds=10, shuffle=True, random_state=None)
    for train_index, test_index in f10:
        x_train, x_test = Train_set_x.ix[train_index,:], Train_set_x.ix[test_index,:]
        y_train, y_test = Train_set_y[train_index], Train_set_y[test_index]
        x_train=x_train.dropna()
        x_test=x_test.dropna()
        y_test=y_test.dropna()
        y_train=y_train.dropna()
        lr.fit(x_train,y_train)
        error=sqrt(np.mean((lr.predict(x_test) - y_test) ** 2))
        print error
        results.append(error)
    plt.plot(test_times,results,label=('RMSE of Linear Regression-Workflow'+str(i)))

    
    

#divide the data into two sets
Train_set_x=df.drop('Size of Backup (GB)',1)
Train_set_y=df.ix[:,'Size of Backup (GB)']
#split the data into 10 folds
f10 = KFold(len(Train_set_x), n_folds=10, shuffle=True, random_state=None)
# result1 linear,result2 poly-2,result3 poly-3,result4 poly-4,result5 poly-5,result6 poly-6,result7 poly-7  
results1 = []
results2 = []
results3 = []
results4 = []
results5 = []
results6 = []
results7 = []
for train_index, test_index in f10:
      X_train, X_test = Train_set_x.ix[train_index,:], Train_set_x.ix[test_index,:]
      y_train, y_test = Train_set_y[train_index], Train_set_y[test_index]
      lr.fit(X_train,y_train)
      error1=sqrt(np.mean((lr.predict(X_test) - y_test) ** 2))
      results1.append(error1)
      
      poly = PolynomialFeatures(degree=2)
      X_train_2=poly.fit_transform(X_train)
      X_test_2=poly.fit_transform(X_test)
      lr.fit(X_train_2,y_train)
      error2=sqrt(np.mean((lr.predict(X_test_2) - y_test) ** 2))
      results2.append(error2)
      
      poly = PolynomialFeatures(degree=3)
      X_train_3=poly.fit_transform(X_train)
      X_test_3=poly.fit_transform(X_test)
      lr.fit(X_train_3,y_train)
      error3=sqrt(np.mean((lr.predict(X_test_3) - y_test) ** 2))
      results3.append(error3)
      
      poly = PolynomialFeatures(degree=4)
      X_train_4=poly.fit_transform(X_train)
      X_test_4=poly.fit_transform(X_test)
      lr.fit(X_train_4,y_train)
      error4=sqrt(np.mean((lr.predict(X_test_4) - y_test) ** 2))
      results4.append(error4)
      
      poly = PolynomialFeatures(degree=5)
      X_train_5=poly.fit_transform(X_train)
      X_test_5=poly.fit_transform(X_test)
      lr.fit(X_train_5,y_train)
      error5=sqrt(np.mean((lr.predict(X_test_5) - y_test) ** 2))
      results5.append(error5)
      
      poly = PolynomialFeatures(degree=6)
      X_train_6=poly.fit_transform(X_train)
      X_test_6=poly.fit_transform(X_test)
      lr.fit(X_train_6,y_train)
      error6=sqrt(np.mean((lr.predict(X_test_6) - y_test) ** 2))
      results6.append(error6)

      poly = PolynomialFeatures(degree=7)
      X_train_7=poly.fit_transform(X_train)
      X_test_7=poly.fit_transform(X_test)
      lr.fit(X_train_7,y_train)
      error7=sqrt(np.mean((lr.predict(X_test_7) - y_test) ** 2))
      results7.append(error7)
#plt.plot(test_times,results1,label='RMSE of Linear Regression')
plt.legend(fontsize=6)
plt.title('Comparsion of Linear Regression between each workflows')
plt.xlabel('Test Times')
plt.ylabel('RMSE')
plt.ylim(0,5e-15)
plt.show()
plt.savefig('problem3-1')  


plt.figure(2)
plt.plot(test_times,results1,label='RMSE of Linear Regression')  
plt.plot(test_times,results2,label='RMSE of Polynomial Regression(Degree 2)')
plt.plot(test_times,results3,label='RMSE of Polynomial Regression(Degree 3)') 
plt.plot(test_times,results4,label='RMSE of Polynomial Regression(Degree 4)') 
plt.plot(test_times,results5,label='RMSE of Polynomial Regression(Degree 5)')  
plt.plot(test_times,results6,label='RMSE of Polynomial Regression(Degree 6)') 
plt.plot(test_times,results7,label='RMSE of Polynomial Regression(Degree 7)') 
plt.legend(fontsize=6)
plt.title('Comparsion of Linear Regression and Polynomial Regression')
plt.xlabel('Test Times')
plt.ylabel('RMSE')
plt.ylim(0.0,0.12)
plt.show()
plt.savefig('problem3-2')     
      
