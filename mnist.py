#!/usr/bin/env python

import tensorflow
from tensorflow.examples.tutorials.mnist import input_data

# Load the MNIST data set
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)

# The basic MLP graph
x = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])
W = tensorflow.Variable(tensorflow.zeros([784, 10]))
b = tensorflow.Variable(tensorflow.zeros([10]))
y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)

# The placeholder for the correct result
real_y = tensorflow.placeholder(tensorflow.float32, [None, 10])

# Loss function
cross_entropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(
    real_y * tensorflow.log(y), axis=[1])
)

# Optimization
optimizer = tensorflow.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(cross_entropy)

# Initialization
init = tensorflow.global_variables_initializer()

epochs = 1000
with tensorflow.Session() as session:
    session.run(init)

    for _ in range(epochs):
        batch_x, batch_y = mnist_data.train.next_batch(100)

        session.run(train_step, feed_dict={x: batch_x, real_y: batch_y})

    correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(real_y, 1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

    network_accuracy = session.run(accuracy, feed_dict={x: mnist_data.test.images, real_y: mnist_data.test.labels})

    print('The accuracy over the MNIST data is {:.2f}%'.format(network_accuracy * 100))
