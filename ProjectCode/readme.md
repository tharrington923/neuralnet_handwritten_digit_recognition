/*******************************************************************************

Instructions for building the code in the ProjectCode directory

*******************************************************************************/

First, login in to CCV. Then, load git using the following command:

$ module load git

Clone the repository from GitHub using:

$ git clone https://github.com/tharrington923/neuralnet_handwritten_digit_recognition

Move to the build directory to build the code.

$ cd ProjectCode/build/

Use the following command to generate the Makefile:

$ cmake CMakeLists.txt

Now, run make.

$ make

The code should build and generate four executables in the ProjectCode/bin folder.

Each of these executables can be run using the standard ./ command. For example,

$ ./testfunctions

runs the testfunctions executable which tests the code to ensure that various functions
work properly.

neuralNet is the executable for the non-threaded version of the code. Note: it is quite slow.
It can take upwards of a few hours to run.

neuralNet_p is the threaded version of the code. It runs significantly faster. Both
the number of neurons in the hidden layer and the stochastic gradient descent batch
size can be changed from the defaults when running by including two numbers. The
first is the number of training epochs. The second is the number of images
in a batch. To run with 15 epochs and a batch size of 1000, run

$ ./neuralNet_p 15 1000

neuralNet_p_printimages can be used to print images from the validation data set.

$ ./neuralNet_p_printimages 10

prints out the first 10 images. The number can be varied to print out the desired number of images.
