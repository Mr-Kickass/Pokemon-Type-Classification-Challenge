# -*- coding: utf-8 -*-
"""
Created on Sat Jan  7 21:09:09 2017

@author: DELL
"""

import tensorflow as tf
import pandas as pd
import numpy as np

pokemonDF=pd.read_csv('Pokemon.csv')
#display first 6 rows of data
pokemonDF.head(6)
pokemonDF.keys()
#define hyperparameters
learning_rate=0.001
epoch=50
batch_size=10
n_classes=18
n_hidden1=512
n_hidden2=256
#define input data and output labels
input_data=pokemonDF[['Total','HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']]
label=pokemonDF['Type 1']
X=tf.placeholder(tf.float32,[None,7])
y=tf.placeholder(tf.float32,[None,n_classes])


input_data_arr=np.array(input_data)
originalY=label.unique()
oneHotLabel=[]
for j in label:
    labelSet=[]
    for i in originalY:
        if j==i:
            labelSet.append(1)
        else:
            labelSet.append(0)
    oneHotLabel.append(labelSet)        
#normalize data

input_data_arr=(input_data_arr-input_data_arr.mean())/input_data_arr.std()
#define training set and test set
trainData=input_data_arr[0:600,:]
testData=input_data_arr[601:801:,]
trainLabel=oneHotLabel[0:600]
testlabel=oneHotLabel[601:801]
    
    
#define weights and biases for a model that consists of
#2 hidden layers
weights= {
          'w1':tf.Variable(tf.random_normal([7,n_hidden1])),
          'w2':tf.Variable(tf.random_normal([n_hidden1,n_hidden2])),
          'out':tf.Variable(tf.random_normal([n_hidden2,n_classes]))    
          }
          
biases = {
          'b1':tf.Variable(tf.random_normal([n_hidden1])),
          'b2':tf.Variable(tf.random_normal([n_hidden2])),
          'out':tf.Variable(tf.random_normal([n_classes]))                                  
          }              

#define model

layer1=tf.matmul(X,weights['w1'])+biases['b1'] 
layer1=tf.nn.relu(layer1)
layer2=tf.matmul(layer1,weights['w2'])+biases['b2']
layer2=tf.nn.relu(layer2)  
out_layer=tf.matmul(layer2,weights['out'])+biases['out'] 
                    
#define cost and optimizer                    

costFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_layer,trainLabel))
#costFunction=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(out_layer,y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(costFunction)
                    
#initialize tensorflow
init=tf.global_variables_initializer()

with tf.Session() as ses:
    ses.run(init)
    for ep in range(epoch):
        avg_loss=0
        for batch_element in range(len(input_data)):
            _,c= ses.run([optimizer,costFunction],feed_dict={X:trainData,y:trainLabel})
            avg_loss+=c
        print("Epoch "+str(ep)+" loss-> " +str(avg_loss/len(input_data)))
        
#evaluate prediction result
    correct_prediction=tf.equal(tf.argmax(out_layer,1),tf.argmax(testlabel,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    
    print("Accuracy: "+str(accuracy.eval({X:testData,y:testlabel}))*100 )
