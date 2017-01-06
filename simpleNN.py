import tensorflow as tf
import pandas as pd
import numpy as np

#read the data
pokemonDF=pd.read_csv('Pokemon.csv')

#print data
print(pokemonDF.head(2))

#define the unique pokemons in the dataset and assign to nclasses
# We expect any pokemon to be in one of the classes determined as below
nclasses=pokemonDF['Type 1'].unique().shape[0]

#define hyperparamters
learning_rate=0.001
training_epochs=200

#network paramters
hidden_1=13
hidden_2=12
input_layer=7
n_classes=18

#define placeholders
X=tf.placeholder(tf.float32,shape=[None,input_layer])
y=tf.placeholder(tf.float32,[None,n_classes])



W = {
     'h1':tf.Variable(tf.random_normal([input_layer,hidden_1])),
     'h2':tf.Variable(tf.random_normal([hidden_1,hidden_2])),
     'out':tf.Variable(tf.random_normal([hidden_2,n_classes]))   
     }
b={
   'b1':tf.Variable(tf.random_normal([hidden_1])),
   'b2':tf.Variable(tf.random_normal([hidden_2])),
   'out':tf.Variable(tf.random_normal([n_classes]))    
   }

hlayer_1=tf.add(tf.matmul(X,W['h1']),b['b1'])
hlayer_1=tf.nn.relu(hlayer_1)
hlayer_2=tf.add(tf.matmul(hlayer_1,W['h2']),b['b2'])
hlayer_2=tf.nn.relu(hlayer_2)
outLayer=tf.add(tf.matmul(hlayer_2,W['out']),b['out'])
cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(outLayer,y))
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate)
train=optimizer.minimize(cost)
init=tf.global_variables_initializer()

with tf.Session() as ses:
    ses.run(init)
    for ep in range(training_epochs):
        ses.run(train)
        if ep %50==0:
            print(ep, ses.run(W), ses.run(b))
            
        