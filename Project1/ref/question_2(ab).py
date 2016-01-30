# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 17:38:34 2016

@author: YuLiqiang
"""




import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import nltk
from math import sqrt
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn import linear_model


#import matplotlib.pyplot as plt
#loggerfile＝'./network_backup_dataset.csv'；
df = pd.read_csv('network_backup_dataset.csv')
#print df[df['Week #']<2]
##plot: x:week # Day of week,start time y: copy size

Train_set_x=df.drop('Size of Backup (GB)',1)
Train_set_y=df.ix[:,'Size of Backup (GB)']

#change the name and flow into integers
name_id=Train_set_x.ix[:,'File Name'].str.split('_')
Train_set_x.ix[:,'File Name']=name_id.str[1]
Train_set_x.ix[:,'Work-Flow-ID'] = Train_set_x.ix[:,'Work-Flow-ID'].str.split('_').str[2]

day_name = Train_set_x.ix[:,'Day of Week'];
day_name = pd.factorize(day_name);
Train_set_x.ix[:,'Day of Week'] = day_name[0]

#using the 10-fold-valiation and use the average predict result.
f10 = KFold(len(Train_set_x), n_folds=10, shuffle=True, random_state=None)
lr = linear_model.LinearRegression()
#random forest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators = 20,criterion = 'mse',max_depth = 8, max_features='auto',random_state=33)


results1 = []
results2 = []
for train_index, test_index in f10:
      X_train, X_test = Train_set_x.ix[train_index,:], Train_set_x.ix[test_index,:]
      y_train, y_test = Train_set_y[train_index], Train_set_y[test_index]
      lr.fit(X_train,y_train)
      rfr = rfr.fit(X_train,y_train)
      print "Lr准确率为：{:.2f}".format(lr.score(X_test,y_test))
      print "RFR准确率为：{:.2f}".format(rfr.score(X_test,y_test))
      error1=sqrt(np.mean((lr.predict(X_test) - y_test) ** 2))
      error2 = sqrt(np.mean((rfr.predict(X_test) - y_test) ** 2))
      print rfr.feature_importances_
      results1.append(error1)
      results2.append(error2)

scores1 = cross_validation.cross_val_score(lr,Train_set_x, Train_set_y, cv=10)
scores2 = cross_validation.cross_val_score(rfr,Train_set_x, Train_set_y, cv=10)
test_times = range(0,10)

plt.figure()
plt.plot(test_times,results1,label='RMSE of Linear Regression')  
plt.plot(test_times,results2,label='RMSE of Random Forest Regression') 
plt.legend()
plt.title('RMSE of Linear Regression v.s. Random Forest Regression')
plt.xlabel('Test Times')
plt.ylabel('RMSE')
plt.ylim(0.0,0.12)
plt.show()
plt.savefig('RMSE_LR_vs_RFR')
#output the results
print "root mean squre error of LR: " + str( np.array(results1).mean() )    
print "root mean squre error of RFR: " + str( np.array(results2).mean() ) 