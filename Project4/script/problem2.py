# -*- coding: utf-8 -*-
"""
Created on Wed Mar  9 13:07:37 2016

@author: YuLiqiang
"""

import numpy as np
from sklearn import linear_model
import math

data = np.loadtxt('problem2_data')
source = data[0:len(data)-1,1:]
target = data[1:,0]