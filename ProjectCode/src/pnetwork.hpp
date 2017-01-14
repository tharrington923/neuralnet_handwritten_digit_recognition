#ifndef PNETWORK_HPP_
#define PNETWORK_HPP_

#include <mutex>  
#include "network.hpp"
#include "network.cpp"

/* This class inherits from the network class and overrides class methods to
   allow the computation to be multithreaded, improving performance. */
class pNetwork: public Network {
public:

  /* Variables to store the gradient for each image in the batch.
     This is a member variable to avoid passing references to multiple threads.*/
  bVector del_b;
  wVector del_w;

  /* Lock to protect the two member variables above from concurrent writing */
  std::mutex delLock;

  /* Network constructor */
  pNetwork(std::vector<int> sizes);

  /* Threaded version of SGD */
  void stochastic_gradient_descent(int startIndex, int endIndex, double eta, int numCores);

  /* Backpropagation algorithm to run the range on a single thread */
  void backprop_batch(int startIndex, int endIndex);

  /* Train network method to use threaded methods */
  void train_network(int numEpochs, int batchSize, double eta, bool test, int numCores);

};

#endif
