# neuralnet_handwritten_digit_recognition

A project containing C++ code for a neural network. 

Handwritten digit recognition is the application of software to identify digits contained in an image. It is used in many contexts in contemporary daily life, including mobile check deposits and handwritten document scanning. This technology is readily extended to handwritten letter recognition and the recognition of strings of numbers and letters as well. This project represents a digestible investigation into how neural networks are implemented. The software was written using traditional neural network handwritten digit recognition algorithms, rather than any pre-existing software libraries for neural network applications. 

Specifically, the neural networks documented herein are modeled after the algorithms presented in Chapters 1 and 2 of Neural Networks and Deep Learning (http://neuralnetworksanddeeplearning.com/). The Python code and mathematical methods provided in that text served as a guide in the design of our neural network code, which I have built from the ground-up in C++.

A directed effort was made to avoid using any references to neural networks already produced using C or C++. Working from a basic neural network, which trains in a single CPU thread, parallelization was implemented in the form of CPU multi-threading. 

The neural networks were trained using the MNIST Database of handwritten digits (http://yann.lecun.com/exdb/mnist/), which comprises a training set of 60,000 images and a test set with 10,000 images. 

The goal of this project was to learn about machine learning and achieve computer digit recognition by building a neural network from scratch in C++. 

Using basic neural networks, digit recognition software with a high success rate (>90%) can be accomplished in a short period of time.

# Code Description

The network code consists of all the code used to create and train a neural network. In the code repository, the code in the ProjectCode directory will be referred to as network code. The network code is composed of two C++ classes, class methods, and helper functions. The fully functioning network code consists of numerous header (*.hpp), and *.cpp files that groups related methods together. 

My neural network is implemented as an object of a custom Network class. The Network class was designed to be used to represent a neural network. The Network class code can be found in network.cpp and network.hpp. 

The figure below shows the UML diagram for the classes used in the network code.

![](https://github.com/tharrington923/neuralnet_handwritten_digit_recognition/blob/master/UMLnetwork.jpg)

The figure below contains the global helper functions used in the Network class.

![](https://github.com/tharrington923/neuralnet_handwritten_digit_recognition/blob/master/helperFunction.jpg)

A neural network is defined by the number of input nodes, the number of hidden layers, the number of nodes in each hidden layer, and the number of nodes in the final layer. For this work, we have chosen a network with a single hidden layer consisting of fifteen nodes. The input layer, defined by the dimensionality of the MNIST data, has 784 (28*28) nodes. The output layer is a vector with 10 components. The output layer is a 10 component vector that represents the greyscale values of the output of the network. The component that contains the pixel with the highest greyscale determines the number. 

A valid neural network is initialized with weight matrices and bias vectors coming from a normal gaussian distribution. Then, the training images are used to train the network's weights and biases using a method of stochastic gradient descent to minimize the cost function. The network is trained for a specified number of epochs and batch size. An epoch is the single use of the entire training data. The batch size is the number of images used to determine the gradient before updating the network's weights and biases. If the batch size was selected as 1000, the network parameters would be updated 60 times each epoch. If a batch size of 60,000 was selected, the network parameters would be updated one time each epoch.

In the standard Network class training procedure, the gradient for each image in a batch is computed sequentially. Then, the average gradient is calculated and the network's parameters are updated. As a result, training the network is a slow process because the gradient is calculated for each training image (there are 60,000 of these) each epoch. Training the network for only 10 epochs would require calculating the gradient 600,000 times! This does not even include any of the time required to update the network parameters after each batch is processed. Training a neural network with 30 nodes in the hidden layer using a batch size of 10 can take upwards of five minutes per epoch. To circumvent this prohibitively slow training time, a threaded version of the code was written to parallelize the code.

The pNetwork (p used to represent parallelized) class inherits from the Network class as seen in the UML class diagram above. The virtual methods of the Network class are overwritten in the pNetwork class to utilize threads. The threaded version of stochastic gradient descent distributes the images in each mini-batch to the maximum number of threads available on the system (typically 2x the number of cores). This allows the mini-batch calculations to be performed sequentially. All of the threads update a common gradient variable in each the mini-batch, but the shared variable is locked whenever one of the threads is updating the value. The variable is only locked for a short time while writing, so the threaded code overall is significantly faster than the non-threaded code. When the mini-batch size is larger than the number of threads, the stochastic gradient descent calculations can see up to a factor of 32 speed up when run on a 16 cores machine.

The network class has many useful features to work with the class. After the network is trained, the weights and biases for each layer can be written to a binary file using the save_network_parameters() method. There is a load_network_parameters() method to load trained network parameter values. The class also has a method to print the loaded MNIST images.

# Testing Code

Confidence in the correctness of the neural network code was established by integrating testing directly into the design of the software. At each stage of the development process, testing was performed. In the network code, C++ asserts were used frequently and can be seen in many of the Network class methods. All of the matrix and vector functions contain assert statements to ensure that the data objects being passed have the correct dimensionality for the matrix/vector function. Arguments to methods are checked to verify that the basic assumptions about the input parameters are true. By using asserts to confirm that each method receives and outputs the desired data structure, bugs were quickly identified. 

Beyond testing individual elements methods of the code, the entire network code used to train the neural network was tested to ensure that the all of the methods were compatible. To test the correctness of the stochastic descent algorithm, a network was initialized with test values and the output of the network after stochastic gradient descent was compared to the expected result from an equivalent Python program. We have included an executable within the ProjectCode/bin directory that tests that the network code passes forward propagation, back propagation, loading/saving network parameters, reading image/label data, and maximum index finding algorithms passes the written tests.

As a final test of the correctness of the trained neural network, the ability of the neural network to identify handwritten digits can be tested using the MNIST test set of handwritten digits. The ability to recognize these digits is a quantitative measure of the network's capabilities because the test set consists of images that were not used to train the network. 

To visualize the MNIST handwritten digit images, a method was written to load the training MNIST image data and print the values of each of the pixels. The executable neuralNetpprintimages is located in the ProjectCode/bin directory. To represent the image pixel data, the greyscale pixel values below 0.7 were replaced with zero and those higher were replaced with one. Two sample printouts of the executable are shown below: First the number 2 and then the number 7.
 
![](https://github.com/tharrington923/neuralnet_handwritten_digit_recognition/blob/master/2.png)
![](https://github.com/tharrington923/neuralnet_handwritten_digit_recognition/blob/master/7.png)

