# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np

plt.rcParams['figure.figsize'] = (10.0, 10.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

## ============= hyperparams starts ====================================== 
training_rate   = 0.01   
training_epochs = 40
batch_size      = 100
l2_reg          = 0.003
n_hidden        = 30     # hidden layer neuron number

## Do not change
n_input         = 2      # input dimension
n_classes       = 3      # output dimension
## ============= hyperparams ends ======================================

## ============= data preparation starts =================================
# read data, 3 class, each class has 300 samples
df = pd.read_csv('spiralData.csv')
points = df.loc[:,['X','Y']]
labels = df.loc[:,['Class']]

# shuffle data and split data into training, validation and test data
df_shuffled = df.sample(frac=1).reset_index(drop=True)
trainData = df_shuffled.loc[:599,['X','Y']]
trainLabel = df_shuffled.loc[:599,['Class']]
valData = df_shuffled.loc[600:799,['X','Y']]
valLabel = df_shuffled.loc[600:799,['Class']]
testData = df_shuffled.loc[800:,['X','Y']]
testLabel = df_shuffled.loc[800:,['Class']]

# convert label into one-hot
length_tmp = len(trainLabel)
trainLabel_onehot = np.zeros((length_tmp,n_classes))
trainLabel_onehot[range(length_tmp),list(trainLabel.transpose().values)] = 1

length_tmp = len(testLabel)
testLabel_onehot = np.zeros((length_tmp,n_classes))
testLabel_onehot[range(length_tmp),list(testLabel.transpose().values)] = 1

length_tmp = len(valLabel)
valLabel_onehot = np.zeros((length_tmp,n_classes))
valLabel_onehot[range(length_tmp),list(valLabel.transpose().values)] = 1

## ============= data preparation ends =================================

## ============= neural network starts =================================
# construct neural network
points = tf.placeholder(tf.float32, [None, n_input])
label = tf.placeholder(tf.float32, [None, n_classes])

# first hidden layer
W1 = tf.Variable(tf.random_normal([n_input, n_hidden]))
b1 = tf.Variable(tf.random_normal([n_hidden]))
# output layer
W2 = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b2 = tf.Variable(tf.random_normal([n_classes]))
hidden_layer = tf.nn.relu(tf.add(tf.matmul(points, W1), b1)) # points * W + b
output_layer = tf.matmul(hidden_layer, W2) + b2

# setup loss and optimizer
softmax_result = tf.nn.softmax_cross_entropy_with_logits_v2(labels=label, logits=output_layer)
l2_loss = l2_reg*tf.nn.l2_loss(W1) + l2_reg*tf.nn.l2_loss(b1) + l2_reg*tf.nn.l2_loss(W1) + l2_reg*tf.nn.l2_loss(b2)
cross_entropy = tf.reduce_mean(softmax_result)
optimizer = tf.train.GradientDescentOptimizer(training_rate)
train_step = optimizer.minimize(cross_entropy)

## ============= neural network ends =================================


## ============= training and test starts ============================
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    
    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(trainData/batch_size))
        
        # Loop over all batches
        for i in range(total_batch):
            offset = 0
            batch_x = trainData[offset:offset+batch_size]
            batch_y = trainLabel_onehot[offset:offset+batch_size]
            offset += batch_size
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([train_step, cross_entropy],
                            feed_dict={points: batch_x, label: batch_y})
            #print("D.chaos cost {}/{}".format(c,avg_cost))
        
            # Compute average loss
            avg_cost += c / total_batch
        
        # accuracy on validation data
        correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        # Display logs per epoch step
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))   
        print("Accuracy on training data: %4f" % accuracy.eval({points: trainData, label: trainLabel_onehot}),end=' ')
        print(", on validation data: %4f" % accuracy.eval({points: valData, label: valLabel_onehot}))
    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(output_layer, 1), tf.argmax(label, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy on test data: %4f" % accuracy.eval({points: testData, label: testLabel_onehot}))

## ============= training and test ends ============================