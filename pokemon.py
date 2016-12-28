## -*- coding: utf-8 -*-
"""
#Created on Wed Dec 28 16:50:28 2016
#
#@author: Ashwin
#"""



from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
import pandas as pd
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.layers.normalization import BatchNormalization

seed=7
np.random.seed(seed)
df=pd.read_csv('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv')
dataset=np.array(df)
uniqueLabel=df['Type 1'].unique()
#define training set and testset
trainSet=dataset[0:801,4:11]
testSet=dataset[560:801,4:11]
compareLabel=dataset[560:801,2:3]
#define labels = 'type 1' column of dataset 
label=dataset[0:801,2:3]

#use label encoder for encoding output
encoder=LabelEncoder()
encoder.fit(label)
encoded_Y=encoder.transform(label)
encodedLabel=np_utils.to_categorical(encoded_Y)
print(encoded_Y.shape)
print(encodedLabel.shape)

#define model and stack layers on top
model=Sequential()
model.add(Dense(512,input_dim=7,init='uniform',activation='relu'))
model.add(Dense(256,init='uniform',activation='relu'))
#attach normalization for output
model.add(Dense(18,init='uniform'))
model.add(BatchNormalization())
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainSet,encodedLabel,nb_epoch=500,batch_size=10)

eval=model.evaluate(trainSet,encodedLabel)
print("%s: %.2f%% " % (model.metrics_names[1], eval[1]*100))

#prediction chunk
prediction=model.predict_classes(testSet)
print(encoded_Y.shape)
print(encodedLabel.shape)
print(prediction)
print(uniqueLabel[prediction])
print("------------------------------------------------------")
