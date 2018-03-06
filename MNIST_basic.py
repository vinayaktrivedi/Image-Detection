import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
import matplotlib.pyplot as plt
x=tf.placeholder(tf.float32,shape=[None,784])
w=tf.Variable( tf.zeros([784,10]))
b=tf.Variable(tf.zeros([10]))
y=tf.matmul(x,w)+b
y_true=tf.placeholder(tf.float32,[None,10])
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true,logits=y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.5)
train=optimizer.minimize(cross_entropy)
init=tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    for step in range(1000):
        batch_x, batch_y =mnist.train.next_batch(100)
        sess.run(train,feed_dict={x:batch_x,y_true:batch_y})
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    print(sess.run(accuracy,feed_dict={x:mnist.test.images,y_true:mnist.test.labels}))
    x_eval=tf.placeholder(tf.float32,shape=[1,784])
    y_eval=tf.matmul(x_eval,w)+b
    answer=tf.argmax(y_eval,1)
    var=mnist.train.images[10].reshape(1,784)
    print(sess.run(answer,feed_dict={x_eval:var}))
    single=mnist.train.images[10].reshape(28,28)
    plt.imshow(single)
    plt.show()
