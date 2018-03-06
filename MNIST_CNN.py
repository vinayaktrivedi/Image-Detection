import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
def init_weights(shape):
    init_random_dist=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init_random_dist)
def init_bias(shape):
    init_bias=tf.constant(0.1,shape=shape)
    return tf.Variable(init_bias)
def convultion1(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
def convolutional_layer(input_x,shape):
    W= init_weights(shape)
    b=init_bias([shape[3]])
    return tf.nn.relu(convultion1(input_x,W)+b)
def normal_full_layer(input_layer,size):
    input_size=int(input_layer.get_shape()[1])
    W=init_weights([input_size,size])
    b=init_bias([size])
    return tf.matmul(input_layer,W)+b
x=tf.placeholder(tf.float32,shape=[None,784])
y_true=tf.placeholder(tf.float32,shape=[None,10])
x_image=tf.reshape(x,[-1,28,28,1])
convo1=convolutional_layer(x_image,[5,5,1,32])
convo1_pooling=max_pool(convo1)
convo2=convolutional_layer(convo1_pooling,shape=[5,5,32,64])
convo2_pooling=max_pool(convo2)
convo2_flat=tf.reshape(convo2_pooling,[-1,7*7*64])
full_layer_one=tf.nn.relu(normal_full_layer(convo2_flat,1024))
hold_prob=tf.placeholder(tf.float32)
full_one_dropout= tf.nn.dropout(full_layer_one,keep_prob=hold_prob)
y_pred=normal_full_layer(full_one_dropout,10)
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y_pred))
optimizer=tf.train.AdamOptimizer(learning_rate=0.001)
train=optimizer.minimize(cross_entropy)
init=tf.global_variables_initializer()
steps=1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(steps):
        batch_x,batch_y=mnist.train.next_batch(50)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.5})

    for i in range(10,100):
        x_eval=mnist.train.images[i].reshape(1,784)
        answer=tf.argmax(y_pred,1)
        print(sess.run(answer,feed_dict={x:x_eval,hold_prob:1.0}))
