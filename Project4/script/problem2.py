# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:07:37 2016

@author: YuLiqiang
"""

import numpy as np
from sklearn import linear_model
import math


hashtag = ['gohawks', 'gopatriots', 'nfl', 'patriots', 'sb49', 'superbowl']
for i in range(len(hashtag)):
    filename = 'problem3_data_#%s' %hashtag[i]
    data_tmp = np.loadtxt(filename)
    if(i == 0):
        data = data_tmp
    else:
        data = np.concatenate((data,data_tmp))
    
source = data[0:len(data)-1,1:]
target = data[1:,0]