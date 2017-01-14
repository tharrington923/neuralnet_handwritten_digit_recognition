#ifndef PNETWORK_CPP_
#define PNETWORK_CPP_

#include <mutex>          // std::mutex
#include <thread>
#include "pnetwork.hpp"

// Constuctor that specifies the number of neurons in each layer
pNetwork::pNetwork(std::vector<int> sizes) {
  msizes = sizes;
  num_layers = msizes.size();
  int seed = 1;
  // Knuth_b random number generator
  std::knuth_b generator(seed);
  // Normal distribution with mean 0.0 and a std dev 1.0
  std::normal_distribution<double> distribution(0.0,1.0);

  mBiases_ = zeroNetworkBiasVector(msizes);
  mWeights_ = zeroNetworkWeightVector(msizes);

  assert(mBiases_.size() == num_layers-1);
  assert(mWeights_.size() == num_layers-1);

  /* Assign random values from a Gaussian distribution with mean 0 and a
     variance 1 to the biases and weight matrices for each neuron in each layer */
  for(int i = 0; i < mBiases_.size(); i++){
    for(int j = 0; j < mBiases_[i].size();j++){
      mBiases_[i][j] = distribution(generator);
      for(int k = 0; k < mWeights_[i][j].size(); k++){
        mWeights_[i][j][k] = distribution(generator);
      }
    }
  }
}

/* Method to calculate the total gradient for a batch of images in the range
  [startIndex,endIndex). Note that startIndex is included and endIndex isn't */
void pNetwork::backprop_batch(int startIndex, int endIndex){

  std::pair<bVector,wVector> nablaPair;

  /* Loop over the indices assigned to the thread this is called in.
     It calculates the gradient for each image and modifies the total gradient.
     The shared variable is locked while writing to it to protect against
     the threads writing to it concurrently. */
  for(int i = startIndex; i < endIndex; i++){
    // Gradient for image i
    nablaPair = backprop(trainingData_[i].first,trainingData_[i].second);
    assert(nablaPair.second.size() == del_w.size());

    // Lock shared variables del_w and del_b and update total gradient
    delLock.lock();
    for(int i = 0; i < nablaPair.second.size(); i++){
      assert(nablaPair.second[i].size() == del_w[i].size());
      assert(nablaPair.first[i].size() == del_b[i].size());
      for(int j = 0; j < nablaPair.second[i].size(); j++){
        del_b[i][j] = del_b[i][j] + nablaPair.first[i][j];
        for(int k = 0; k < nablaPair.second[i][j].size(); k++){
          del_w[i][j][k] = del_w[i][j][k] + nablaPair.second[i][j][k];
        }
      }
    }
    // Release the lock
    delLock.unlock();
  }
};
 /* Perform gradient descent for a batch of images in the range of indices given
    The calculation is divided into the maximum number of allowed concurrent
    threads.  */
void pNetwork::stochastic_gradient_descent(int startIndex, int endIndex, double eta, int numCores){
  assert(endIndex > startIndex);
  assert(startIndex >= 0);

  std::vector<std::thread> threadVector;
  int batchSize = (endIndex - startIndex);

  int numJobsPerThread = batchSize/numCores+1;
  int numThreads = batchSize/numJobsPerThread;
  //std::cout << "Number of threads = " << numThreads << std::endl;
  //std::cout << "Number of backprop per thread = " << numJobsPerThread << std::endl;

  // Split the job into number of allowed threads.
  for(int n = 0; n < numThreads*numJobsPerThread; n=n+numJobsPerThread){
    threadVector.push_back(std::thread(&pNetwork::backprop_batch,this,startIndex+n,startIndex+n+numJobsPerThread));
  }
  /* Assigning the appropriate number of jobs to the final thread (will only
     contain the same number as all the others if the jobs can be evenly divided
     into the number of available threads.)*/
  int numProcessed = (numThreads*numJobsPerThread);
  if(numProcessed < batchSize){
    threadVector.push_back(std::thread(&pNetwork::backprop_batch,this,startIndex+numProcessed,endIndex));
  }

  // Waiting for all the threads to finish
  for(int n = 0; n < threadVector.size(); n++){
    if(threadVector[n].joinable()){
      threadVector[n].join();
    }
  }

  /* Average the total gradient and update the network parameters */
  double mu = eta/batchSize;

  //Now we update the weights and biases
  assert(mWeights_.size() == del_w.size());
  for(int i = 0; i < mWeights_.size(); i++){
    assert(mWeights_[i].size() == del_w[i].size());
    assert(mBiases_[i].size() == del_b[i].size());
    for(int j = 0; j < mWeights_[i].size(); j++){
      mBiases_[i][j] = mBiases_[i][j] - (mu*del_b[i][j]);
      for(int k = 0; k < mWeights_[i][j].size(); k++){
        mWeights_[i][j][k] = mWeights_[i][j][k] - (mu*del_w[i][j][k]);
      }
    }
  }

};

/* Method to train the network  */
void pNetwork::train_network(int numEpochs, int batchSize, double eta, bool test, int numCores){
  assert(batchSize > 0);

  //bVector imageData = read_MNIST_image_data("../../MNIST_Data/train-images-idx3-ubyte");
  bVector imageData = read_MNIST_image_data_BW("../../MNIST_Data/train-images-idx3-ubyte",0.5);
  dVector labelData = read_MNIST_label_data("../../MNIST_Data/train-labels-idx1-ubyte");

  trainingData_ = pair_image_label_data(imageData,labelData);
  assert(batchSize <= trainingData_.size());

  int numBatches = trainingData_.size()/batchSize;
  assert(numBatches*batchSize-1< trainingData_.size());

  bVector testImageData;
  dVector testLabelData;
  std::vector<std::pair<dVector,dVector> > testData;
  if(test){
    testImageData = read_MNIST_image_data("../../MNIST_Data/t10k-images-idx3-ubyte");
    testLabelData = read_MNIST_label_data("../../MNIST_Data/t10k-labels-idx1-ubyte");
    testData = pair_image_label_data(testImageData,testLabelData);
  }
  for(int j = 0; j < numEpochs; j++){
    // Shuffle trainingData
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::shuffle(trainingData_.begin(),trainingData_.end(),std::default_random_engine(seed));

    int start, end;
    for(int i = 0; i < numBatches; i++){
      start = i*batchSize;
      end = (i+1)*batchSize; //removed -1
      if(end > trainingData_.size()){
        end = trainingData_.size();
      }
      assert(end <= trainingData_.size());
      del_b = zeroNetworkBiasVector(msizes);
      del_w = zeroNetworkWeightVector(msizes);
      stochastic_gradient_descent(start,end,eta,numCores);
    }
    if(test){
      std::cout << "Epoch " << j << ": " << test_network(testData) << "/" << testLabelData.size() << std::endl;
    }
    else {
      std::cout << "Epoch " << j << " complete." << std::endl;
    }
  }
};


#endif
