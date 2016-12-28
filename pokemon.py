# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 16:50:28 2016

@author: DELL
"""

import numpy as np
import pandas as pd
import tfnet
#import tensorflow as tf
df=pd.read_csv('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv')
#print(df.keys())
#print(df.head(1))
#df=df[:,4:9]
arr=np.array(df)
#print(df['Type 1'].unique()) 
#print(len(arr[:,4:9]))
labels=arr[:,2:3]
var=arr[:,4:11]
print("------------")
#print(len(arr[:,2]))
uniqueClass=df['Type 1'].unique() 
dataset=np.hstack([var,labels])
print(uniqueClass)

data = np.asarray(var,dtype=np.float32)
mean = data.mean(axis=0)
std = data.std(axis=0)
data = (data - mean) / std

#create 70% train data
nrow=df.shape[0]
print(data[0:1,:])
train_data=data[0:561,:]
train_labels=labels[0:561]
test_data=data[562:801,:]
test_labels=labels[562:801]
tfnet.main(train_data, train_labels, test_data, test_labels)
	