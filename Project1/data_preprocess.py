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
target_forgot=network.values[:,6:7]

##transform the origin date into the proper forms
##the final result is in prepossessed
##meaning of its columns
##0:15 week #
##15:22 weekdays
##22:28 startTime
##28:33 workflow
##33:63 fileName
##63:64 Backup Time(hour)
weeks=network.values[:,0:1]
bweeks=np.zeros((18588,15))
i=0
for tmp in np.nditer(weeks):
    bweeks[i,int(tmp)-1]=1
    i+=1

weekdays=network.values[:,1:2]
bweekdays=np.zeros((18588,7))
i=0
for tmp in np.nditer(weekdays):
    bweekdays[i,int(tmp)-1]=1
    i+=1
    
startTimes=network.values[:,2:3]
bstartTimes=np.zeros((18588,6))
i=0
for tmp in np.nditer(startTimes):
    bstartTimes[i,int(tmp)/4]=1
    i+=1
    
workflows=network.values[:,3:4]
bworkflows=np.zeros((18588,5))
i=0
for tmp in np.nditer(workflows):
    bworkflows[i,int(tmp)]=1
    i+=1
    
fileNames=network.values[:,4:5]
bfileNames=np.zeros((18588,30))
i=0
for tmp in np.nditer(fileNames):
    bfileNames[i,int(tmp)]=1
    i+=1
    
prepossessed=np.concatenate((bweeks,bweekdays,bstartTimes,bworkflows,bfileNames,target_forgot),axis=1)



#print(lr.coef_)
#print target.max()
#plt.scatter(target, lr.predict(data))
#plt.plot([0, 1],[0, 1], 'k--', lw = 4)
#plt.show()


