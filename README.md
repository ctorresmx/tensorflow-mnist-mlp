# TensorFlow 101

I have been fiddling with Machine Learning since I went to college, I implemented a couple of neural networks from scratch using C, and building them from zero is not a walk in the park. I am beginning to learn this framework for a very simple reason, it is readily available, you don't need to worry about the low-level implementation details.

I am not saying that building your own neural network in C is hard, but it consumes time that you could well spend tweaking your model, acquiring training data and ultimately watching your neural network classify, predict or whatever you trained it for.

In this post I want to show you how to quickly use MNIST data set to train a MLP (Multi-layer perceptron) model. I am not going to show you any visual data, or use concepts from Deep Learning and/or CNNs (Convolutional Neural Networks), but take this as a brief introduction.

Much of what I am writing here is my take on the book "Hands-On Machine Learning with Scikit-Learn and TensorFlow" by Aurélien Géron. It is a great book that I am enjoying a lot. Also, I used TensorFlow's examples as a guide.

TensorFlow provides an easy way to just worry about your model, your data and your result. It was developed by Google's Brain team and it is used internally in a lot of their products, also Alphabet's DeepMind migrated over from Torch. This, I think, is enough evidence that TensorFlow is a great framework for Deep Learning.

According to TensorFlow's [official page](tensorflow.org):

> TensorFlow™ is an open source software library for numerical computation using data flow graphs. Nodes in the graph represent mathematical operations, while the graph edges represent the multidimensional data arrays (tensors) communicated between them.

TensorFlow's main purpose is to work as a framework for deep learning and neural networks research, although the fact that it bases itself in data flow graphs and tensors make it easy to be useful in other areas where we can represent the mathematical operations as a graph.

The framework is composed of two main parts, the data flow graphs and the tensors. The data flow graph is basically a flow diagram, we define the mathematical operations on tensors and their relationship towards other operations. We can think of a tensor as a multi-dimensional matrix. So, if we want to create a basic graph that multiples two matrices then TensorFlow can be used for it.

## Installation

TensorFlow has the possibility of running on a CPU or a GPU. Obviously GPU is the fastest option, but since I don't have a machine with a CUDA-enabled graphics card, then I will just go around with the CPU installation.

Python's package manager, pip, provides by far the easiest way to install.

```bash
$ pip install tensorflow
```

Everything is going to be handled for you, the downside is that pip installs the most general TensorFlow package, meaning that you won't get any possible CPU optimizations from compiling with the specifics of your processor chip, for that you will need to compile your own TensorFlow. But it is enough to try the basics of the framework.

Now, to test that everything was installed correctly, you can run this simple snippet in Python's interactive shell.

```python
>>> import tensorflow
>>> hello = tensorflow.constant('Hello, TensorFlow!')
>>> sess = tensorflow.Session()
>>> print(sess.run(hello))
```

Then you will see a nice `Hello, TensorFlow!` if everything worked great.

```python
b'Hello, TensorFlow!'
```

## TensorFlow basics

TensorFlow has a basic set of concepts which will be useful to build our graphs.

- Placeholder
- Variables
- Loss function
- Optimization
- Session

### Placeholder

Think of a placeholder as a reserved space for data we are going to input dynamically, e.g., the inputs of a neural network will be a placeholder, specially when we are making predictions, we never know in advance what are we going to introduce to the model.

### Variables

These variables will typically hold the weights and biases, by default, TensorFlow will modify variables in order to train the network, but there is an option to set this off so a variable is never modified by the training session.

### Loss function

Loss functions determine the performance of the network, we can pretty much define it in any way possible, but in order to train a network, we have to make this one clear and useful towards determining what is good or bad performance. TensorFlow provides a lot of basic mathematical functions to create the loss function (logarithms, mean, sum, etc).

### Optimization

TensorFlow provides a whole range of optimizers, you have Gradient Descent, Adam, Momentum, AdaGrad, etc. Each one of them have their own quirks and perks. These are the ones that determine how your loss function is going to be minimized and in turn your network trained.

### Session

A TensorFlow session is basically the environment were your graph runs, it performs all the defined operations.

Now that, what I think are the basics, are covered, it is time to go into code.

## Training a MLP using MNIST data set

Let's start by importing the libraries we are going to use. We need TensorFlow (obviously) and we need to import the MNIST data set helper function.

TensorFlow includes a range of sample data to play with, but for this one let's play with MNIST data. The MNIST data set is a well-known collection of handwritten digits, it contains 60k training images and 10k testing images. Each image has a size of 28x28 pixels. It has been used for a long time as the `Hello, World!` of Machine Learning.

```python
import tensorflow
from tensorflow.examples.tutorials.mnist import input_data
```

The following line will download the data set and put it under our working directory. We use the `one_hot` flag to tell TensorFlow that we are using one-hot encoded vectors, i.e., every number from the MNIST data set will be encoded as an array of ones and zeros.

```python
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
```

A neural network is basically a function like:

```
y = f((W * x) + b)
```

Where:

- `x`: Input vector
- `W`: Weight vector
- `b`: Bias vector
- `f`: Activation function
- `y`: Output vector (prediction)

So, the input vector should be a placeholder; weight vector is a variable as well as the bias vector; the activation function is one of many TensorFlow functions; the output vector is basically the grouping of all of this.

```python
x = tensorflow.placeholder(tensorflow.float32, shape=[None, 784])
W = tensorflow.Variable(tensorflow.zeros([784, 10]))
b = tensorflow.Variable(tensorflow.zeros([10]))
y = tensorflow.nn.softmax(tensorflow.matmul(x, W) + b)
```

Notice the `None` when defining `x`, `None` here means that we don't know in advance the size of the tensor in that dimension, so shape `[None, 784]` means that we are going to input a variable number of vectors of size 784, which is the size of a flatten MNIST image vector (28x28 = 784).

The `tensorflow.zeros()` function creates a vector of the given shape filled with zeros.

Finally, notice that we didn't define the activation function `f`, we just defined the output `y` as the softmax function that takes the formula `(W * x) + b`.

Be aware that we defined `x` as a vector n x 784, while W is a vector of size 784 x 10, this means that when defining the multiplication on `y` the order of multiplication was `x * W`, not the other way around, which is mathematically wrong.

This covers our basic graph. We have just defined how our neural network will flow, but that's not all, if we input a vector right now we won't get our expected results, we need to train the network using our training data set.

To train the neural network we will need to define 3 things: the loss function, the optimization method and the desired output (which comes from the training data).

For this network we are going to use `cross entropy` loss, in simple terms, this function is better suited as a loss function, but for deep lecture on this, please refer to [Michael Nielsen's chapter on cross entropy](http://neuralnetworksanddeeplearning.com/chap3.html#the_cross-entropy_cost_function).

The cross entropy function can defined mathematically as:

```
C = -sum(real_y * ln(predicted_y))
```

Translating that into TensorFlow idiom:

```python
cross_entropy = tensorflow.reduce_mean(-tensorflow.reduce_sum(
    real_y * tensorflow.log(y), axis=[1])
)
```

Note that we added the `reduce_mean` function, which basically extracts the mean value of a given array, in this case of the sum.

After defining the loss function, it is time to define the functions that will actually train the neural network.

```python
optimizer = tensorflow.train.GradientDescentOptimizer(0.5)
train_step = optimizer.minimize(cross_entropy)
```

We use the `GradientDescentOptimizer` which the name is quite descriptive. The `0.5` value corresponds to the learning rate. Finally, we define the training step, this tells TensorFlow what is our ultimate goal, which is obviosly minimizing the cross entropy via the Gradient Descent algorihtm.

So far, so good, we have defined our neural network model, we defined the loss function and the training method, but at this point TensorFlow hasn't done anything yet, this is because TensorFlow actually asks you to design the graph before building and running it.

A Neural Network training session consists of two basic steps, first you input the data and see how badly the network is doing, then you back-propagate the error (from the loss function) while modifying the weights and biases. This is done multiple times with the training data set, until the network converges to a desired error rate or a number of epochs have been elapsed, whichever comes first.

There's nothing stopping us from training until the error rate is 0, but this is hardly achievable, which basically means `impossible`. For the sake of this tutorial let's use an arbitrary number of epochs, let's say 1000.

```python
epochs = 1000
```

Before starting the training, let's initialize the graph, this step will reserve memory for our placeholders, create variables and define our loss functions and optimizers. Remember, none of this has been actually executed on our machine, not until we have our TensorFlow session.

```python
init = tensorflow.global_variables_initializer()
```

The final step is to initialize a TensorFlow session, using Python's context manager is the easier way. Once we have a TensorFlow session we can actually run the initialization, which now will load everything necessary on memory.

```python
with tensorflow.Session() as session:
    session.run(init)

    for _ in range(epochs):
        batch_x, batch_y = mnist_data.train.next_batch(100)

        session.run(train_step, feed_dict={x: batch_x, real_y: batch_y})

    correct_prediction = tensorflow.equal(tensorflow.argmax(y, 1), tensorflow.argmax(real_y, 1))
    accuracy = tensorflow.reduce_mean(tensorflow.cast(correct_prediction, tensorflow.float32))

    network_accuracy = session.run(accuracy, feed_dict={x: mnist_data.test.images, real_y: mnist_data.test.labels})

    print('The accuracy over the MNIST data is {:.2f}%'.format(network_accuracy * 100))
```

For each epoch we need to get a batch of MNIST data. MNIST data set library has a convenient `next_batch()` function which grabs `n` vectors of data. After we got our next batch it's time to run our `train_step`, feeding our input vector `batch_x` and the expected output `batch_y`. Now we see the use of placeholders.

For the last part, we need to measure how well our neural network is doing, for this we define two things: what was the expected prediction and how to measure the accuracy.

MNIST outputs are 10-dimension vectors (numbers from 0 to 9) one-hot encoded (remember at the very beginning, we set `one_hot=True`), this means that a correct prediction for the number `7` will have a vector `0,0,0,0,0,0,0,1,0,0`. Our neural network outputs decimal numbers so we need to round the output. Lastly, the accuracy is nothing more than the mean value across all the given vectors.

Finally, we run the accuracy operation using the test data from the MNIST set. Note that `correct_prediction` depends on the `y` operation and the `real_y` placeholder, hence, this `accuracy` operation will make a forward propagation of the network with the given data.

```bash
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
The accuracy over the MNIST data is 92.02%
```

As we can see we are getting around `92%` accuracy, not bad, but it's not state-of-the-art level, yet...

That's it! Now you have a one-layer MLP network trained on MNIST data, this is a very very small example, but I think it is useful to understand the very basics of TensorFlow.

If you want to download the complete code, get it at my [tensorflow-mnist-mlp](https://github.com/ctorresmx/tensorflow-mnist-mlp) repository.
