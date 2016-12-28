## -*- coding: utf-8 -*-
#"""
#Created on Wed Dec 28 16:50:28 2016
#
#@author: DELL
#"""
#
#import numpy as np
#import pandas as pd
#import tfnet
##import tensorflow as tf
#df=pd.read_csv('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv')
##print(df.keys())
##print(df.head(1))
##df=df[:,4:9]
#arr=np.array(df)
##print(df['Type 1'].unique()) 
##print(len(arr[:,4:9]))
#labels=arr[:,2:3]
#var=arr[:,4:11]
#print("------------")
##print(len(arr[:,2]))
#uniqueClass=df['Type 1'].unique() 
#dataset=np.hstack([var,labels])
#print(uniqueClass)
#
#data = np.asarray(var,dtype=np.float32)
#mean = data.mean(axis=0)
#std = data.std(axis=0)
#data = (data - mean) / std
#
##create 70% train data
#nrow=df.shape[0]
#print(data[0:1,:])
#train_data=data[0:561,:]
#train_labels=labels[0:561]
#test_data=data[562:801,:]
#test_labels=labels[562:801]
#tfnet.main(train_data, train_labels, test_data, test_labels)
#	

from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization

seed=7
np.random.seed(seed)
dataset=np.array(pd.read_csv('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv'))
#dataset=np.loadtxt('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv', delimiter=",")

trainSet=dataset[:,4:11]
label=dataset[:,2:3]

encoder=LabelEncoder()
encoder.fit(label)
encoded_Y=encoder.transform(label)
encodedLabel=np_utils.to_categorical(encoded_Y)

model=Sequential()
model.add(Dense(13,input_dim=7,init='uniform',activation='relu'))
model.add(Dense(8,init='uniform',activation='relu'))

model.add(Dense(18,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainSet,encodedLabel,nb_epoch=100,batch_size=10)

eval=model.evaluate(trainSet,label)
print("%s: %.2f%% " % (model.metrics_names[1], eval[1]*100))
