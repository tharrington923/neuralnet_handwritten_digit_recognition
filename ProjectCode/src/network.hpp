#ifndef NETWORK_H_
#define NETWORK_H_

#include "aliases.hpp"
#include <vector>
#include <string>

class Network {
  public:
//protected:

  /* Number of layers in neural Network */
  int num_layers;

  /* Number of neurons in each layer */
  std::vector<int> msizes;

  /* Neuron biases. Vector of Vectors. One for each neuron in each layer (excluding input) */
  bVector mBiases_;
  /* Neuron weights. Vector of Matrices (represented as Vector of Vectors) */
  wVector mWeights_;

  /* Training data variable */
  std::vector<std::pair<dVector,dVector> > trainingData_;

//public:

  Network(){};

  /* The class constructor that takes a vector<int> sizes
     sizes contains the number of neurons in each layer of the neural network.
     The first layer is assumed to be an input layer, so there are no biases for
     the input layer. */
  Network(std::vector<int> sizes);

  /*  */
  Network(std::vector<int> sizes, bVector& biases, wVector& weights);

  /* Returns the ouput of the network when inputVector is fed into it  */
  dVector feedforward(dVector& inputVector);

  /* Update nabla_b, nabla_w representing the gradient for the cost function C_x */
  /* The nablas are the same dimensions as the respective bias and weight vectors */
  std::pair<bVector, wVector > backprop(dVector& inputVector, dVector& expectedOutput);

  dVector cost_derivative(dVector& output_activations, dVector& y);

  // batchData is the set of x and y stored in a vector of pairs
  //virtual void stochastic_gradient_descent(std::vector<std::pair<dVector,dVector> >& batchData, double eta);
  virtual void stochastic_gradient_descent(int startIndex, int endIndex, double eta);

  virtual void train_network(int numEpochs, int batchSize, double eta, bool test);

  virtual int test_network(std::vector<std::pair<dVector,dVector> >& testData);

  // Method to save the network biases and weights. Saved as a binary file
  void save_network_parameters(std::string filename);

  void load_network_parameters(std::string filename);

};

#endif
