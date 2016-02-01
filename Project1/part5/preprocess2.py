# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:06:29 2016

@author: YuLiqiang
"""


import numpy as np
import pandas as pa


##transform the origin date into the proper forms
##the final result is in prepossessed
def preprocess_2(data):
    rowNO=len(data)
    indices=data[:,8:9]
    bindices=np.zeros((rowNO,9))
    i=0
    for tmp in np.nditer(indices):
        if(tmp==24):
            bindices[i,8]=1;
        else:
            bindices[i,int(tmp)-1]=1
        i+=1
    
        
    preprocessed=np.concatenate((data[:,0:8],bindices,data[:,9:13]),axis=1)
    return preprocessed

#print(lr.coef_)
#print target.max()
#plt.scatter(target, lr.predict(data))
#plt.plot([0, 1],[0, 1], 'k--', lw = 4)
#plt.show()


