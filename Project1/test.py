# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 19:05:09 2016

@author: YuLiqiang
"""
import numpy as np
import scipy as sci
import pandas as pa
import matplotlib.pyplot as plt
def mapWeekToDay(week):
    return (week-1) * 7
    
network = pa.read_csv("network_backup_dataset.csv", header = 0)
dict = {"Monday" : "1", "Tuesday":"2", "Wednesday":"3", "Thursday":"4", "Friday":"5", "Saturday":"6", "Sunday":"7"}

for i in dict:
    network['Day of Week']=[s.replace(i, dict[i]) for s in network['Day of Week']]
network['Day of Week']=[int(s) for s in network['Day of Week']]
network_flow0=network[network["Work-Flow-ID"]=="work_flow_0"]
#network["time"] = 
network_tmp = network_flow0.groupby(["Week #", "Day of Week","Backup Start Time - Hour of Day"])["Size of Backup (GB)"].sum()
network_tmp = network_tmp.values;
plt.plot(network_tmp[0:120])
#toplot = network[network['Week #'] <= 4]
#plt.plot(np.arange(105), network_tmp["Size of Backup (GB)"]);
#plt.plot((toplot["Week #"]-1)*7 + toplot["Day of Week"], toplot["Size of Backup (GB)"], marker = "*", linestyle="None")
plt.show();


