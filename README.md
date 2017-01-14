# neuralnet_handwritten_digit_recognition

A project containing C++ code for a neural network. 

Handwritten digit recognition is the application of software to identify digits contained in an image. It is used in many contexts in contemporary daily life, including mobile check deposits and handwritten document scanning. This technology is readily extended to handwritten letter recognition and the recognition of strings of numbers and letters as well. This project represents a digestible investigation into how neural networks are implemented. The software was written using traditional neural network handwritten digit recognition algorithms, rather than any pre-existing software libraries for neural network applications. 

Specifically, the neural networks documented herein are modeled after the algorithms presented in Chapters 1 and 2 of Neural Networks and Deep Learning (http://neuralnetworksanddeeplearning.com/). The Python code and mathematical methods provided in that text served as a guide in the design of our neural network code, which I have built from the ground-up in C++.

A directed effort was made to avoid using any references to neural networks already produced using C or C++. Working from a basic neural network, which trains in a single CPU thread, parallelization was implemented in the form of CPU multi-threading. 

The neural networks were trained using the MNIST Database of handwritten digits (http://yann.lecun.com/exdb/mnist/), which comprises a training set of 60,000 images and a test set with 10,000 images. 

The goal of this project was to learn about machine learning and achieve computer digit recognition by building a neural network from scratch in C++. 

Using basic neural networks, digit recognition software with a high success rate (>90%) can be accomplished in a short period of time.

