#!/usr/bin/env python

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])

W = tensorflow.Variable(tensorflow.zeros([784, 10]))
b = tensorflow.Variable(tensorflow.zeros([10]))

y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)

y_ = tensorflow.placeholder(tensorflow.float32, [None, 10])

cross_entropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(y_ * tensorflow.log(y), reduction_indices=[1]))

train_step = tensorflow.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

session = tensorflow.InteractiveSession()

tensorflow.global_variables_initializer().run()

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)

    session.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(y_, 1))
accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

print(session.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
prediction = tensorflow.argmax(y, 1)

print(session.run(prediction, feed_dict={x: mnist.test.images[:1]}))
