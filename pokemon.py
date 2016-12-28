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
dataset=np.array(pd.read_csv('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv'))
#dataset=np.loadtxt('C:/Users/DELL/Anaconda3/Datasets/Pokemon.csv', delimiter=",")

trainSet=dataset[0:560,4:11]
testSet=dataset[560:801,4:11]
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

prediction=model.predict(testSet)

roundOff=[round(x) for x in prediction]
print(roundOff)
