# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:06:29 2016

@author: YuLiqiang
"""

import numpy as np
import pandas as pa
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize

network = pa.read_csv("network_backup_dataset.csv", header = 0)
dict = {"Monday" : "1", "Tuesday":"2", "Wednesday":"3", "Thursday":"4", "Friday":"5", "Saturday":"6", "Sunday":"7"}
for i in dict:
    network['Day of Week']=[s.replace(i, dict[i]) for s in network['Day of Week']]
  
network['File Name']=[s.replace("File_","") for s in network['File Name']]
network['Work-Flow-ID']=[s.replace("work_flow_","") for s in network['Work-Flow-ID']]
network['Day of Week']=[int(s) for s in network['Day of Week']]
network['File Name']=[int(s) for s in network['File Name']]
network['Work-Flow-ID'] = [int(s) for s in network['Work-Flow-ID']]

data1 = network.values[:, 0:5]
data2 = network.values[:, 6:7]
data = np.concatenate((data1, data2), axis=1)
#data_norm = normalize(data, axis=0).ravel()
target = network.values[:,5:6]
lr = linear_model.LinearRegression(normalize = True)
lr.fit(data, target)
print sqrt(np.mean((lr.predict(data) - target) ** 2))
#print(lr.coef_)
#print target.max()
plt.scatter(target, lr.predict(data))
plt.plot([0, 1],[0, 1], 'k--', lw = 4)
plt.show()


