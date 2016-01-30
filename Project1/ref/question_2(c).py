# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 21:21:48 2016

@author: Xinxin
"""
import pandas as pd
import matplotlib as plt
import numpy as np
import math
from math import sqrt
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from pybrain.datasets import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.structure import SigmoidLayer, LinearLayer
from pybrain.supervised.trainers import BackpropTrainer
import pylab

df = pd.read_csv('network_backup_dataset.csv')
X=df.drop('Size of Backup (GB)',1)
y=df.ix[:,'Size of Backup (GB)']

#change the name and flow into integers
name_id=X.ix[:,'File Name'].str.split('_')
X.ix[:,'File Name']=name_id.str[1]
X.ix[:,'Work-Flow-ID'] = X.ix[:,'Work-Flow-ID'].str.split('_').str[2]

day_name = X.ix[:,'Day of Week']
day_name = pd.factorize(day_name)
X.ix[:,'Day of Week'] = day_name[0]

f10 = KFold(len(X), n_folds=10, shuffle=True, random_state=None)
ds = SupervisedDataSet(6, 1) #init the training dataset


# build a neural network
# number of input, hidden units and output units, class of hiddenclass and out class
net = buildNetwork(6, 3, 1, bias = True, hiddenclass = SigmoidLayer, outclass = LinearLayer)

results3 = []

for train_index, test_index in f10:
      X_train, X_test = X.ix[train_index,:], X.ix[test_index,:]
      y_train, y_test = y[train_index], y[test_index]
      #build traing dataset for neural network
      for xx, yy in zip(X_train.values, y_train.values):
          ds.addSample(tuple(xx), yy)
      #build trainer 
      trainer = BackpropTrainer(net,ds)
      trainer.train()
      #trainer.trainUntilConvergence(maxEpochs = 10)
      y_predict = []
      for x in X_test.values:
          y_predict.append(net.activate(tuple(x))[0])
      error3 = sqrt(np.mean((y_predict - y_test) ** 2))
      print error3
      results3.append(error3)
      ds.clear()
      
print "root mean squre error of Neural Network: " + str( np.array(results3).mean() )

      

                   

