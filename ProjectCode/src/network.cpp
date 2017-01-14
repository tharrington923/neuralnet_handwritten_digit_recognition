#ifndef NETWORK_CPP_
#define NETWORK_CPP_

#include "aliases.hpp"
#include "network.hpp"
#include "networkfunctions.cpp"
#include "read.cpp"
#include <vector>
#include <cassert>
#include <iostream>
#include <random>
#include <ctime>        // std::time
#include <algorithm>    // std::shuffle
#include <chrono>       // std::chrono::system_clock

Network::Network(std::vector<int> sizes) {
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

Network::Network(std::vector<int> sizes, bVector& biases, wVector& weights) {
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

  assert(mBiases_.size() == biases.size());
  assert(mWeights_.size() == weights.size());
  for(int i = 0; i < mBiases_.size(); i++){
    assert(mBiases_[i].size() == biases[i].size());
    for(int j = 0; j < mBiases_[i].size();j++){
      assert(mWeights_[i][j].size() == weights[i][j].size());
    }
  }

  /* Assign the biases and weights to the assigned weights */
  for(int i = 0; i < mBiases_.size(); i++){
    for(int j = 0; j < mBiases_[i].size();j++){
      mBiases_[i][j] = biases[i][j];
      for(int k = 0; k < mWeights_[i][j].size(); k++){
        mWeights_[i][j][k] = weights[i][j][k];
      }
    }
  }

}

dVector Network::feedforward(dVector& inputVector){
  /* Returns the ouput of the network when inputVector is fed into it  */
  assert(inputVector.size() == msizes[0]);
  dVector outputVector;
  outputVector = inputVector;
  dVector dotVector;
  dVector addVector;
  for(int i = 0; i < mBiases_.size(); i++){
    dotVector.clear();
    addVector.clear();
    dotVector = vector_dot(mWeights_[i],outputVector);
    addVector = vector_add(dotVector,mBiases_[i]);
    outputVector = sigmoid_vector(addVector);
  }
  return outputVector;
}

/* Update nabla_b, nabla_w representing the gradient for the cost function C_x */
/* The nablas are the same dimensions as the respective bias and weight vectors */
std::pair<bVector, wVector > Network::backprop(dVector& inputVector, dVector& expectedOutput){

  std::pair<bVector, wVector > nablaPair;
  bVector nabla_b = zeroNetworkBiasVector(msizes);
  wVector nabla_w = zeroNetworkWeightVector(msizes);

  dVector activation = inputVector;

  // List to store all activations, layer by layer
  bVector activations;
  activations.push_back(activation);

  // List to store all the z vectors
  bVector zs;

  // Vectors needed in the calculation
  dVector z, dotVector;

  for(int i = 0; i < mBiases_.size(); i++){
    dotVector = vector_dot(mWeights_[i],activation);
    assert(dotVector.size()==mBiases_[i].size());
    z = vector_add(dotVector,mBiases_[i]);
    zs.push_back(z);
    activation = sigmoid_vector(z);
    activations.push_back(activation);
  }

  // Backward pass
  dVector delta, sigmoidPrimeVector;

  // Derivative of sigmoid function
  sigmoidPrimeVector = sigmoid_prime_vector(zs[zs.size()-1]);

  delta = cost_derivative(activations[activations.size()-1],expectedOutput);
  delta = vector_multiply(delta,sigmoidPrimeVector);
  assert(nabla_b[nabla_b.size()-1].size() == delta.size());
  nabla_b[nabla_b.size()-1] = delta;

  bVector outerProduct;
  outerProduct = vector_outer_product(delta,activations[activations.size()-2]);
  assert(nabla_w[nabla_w.size()-1].size() == outerProduct.size());
  assert(nabla_w[nabla_w.size()-1][0].size() == outerProduct[0].size());
  nabla_w[nabla_w.size()-1] = outerProduct;

  // Calculate the gradients for the remaining layers. Working from next to
  // last layer to the first layer.
  dVector sp;
  for(int i = nabla_b.size()-2; i >= 0; i--){
    z = zs[i];
    sp = sigmoid_prime_vector(z);
    delta = vector_transpose_product(mWeights_[i+1],delta);
    delta = vector_multiply(delta,sp);
    assert(nabla_b[i].size() == delta.size());
    nabla_b[i] = delta;
    outerProduct = vector_outer_product(delta,activations[i]);
    assert(nabla_w[i].size() == outerProduct.size());
    assert(nabla_w[i][0].size() == outerProduct[0].size());
    nabla_w[i] = outerProduct;
  }
  nablaPair.first = nabla_b;
  nablaPair.second = nabla_w;
  return nablaPair;
}

dVector Network::cost_derivative(dVector& output_activations, dVector& y){
  return vector_subtract(output_activations,y);
}

// batchData is the set of x and y stored in a vector of pairs
// endIndex is not inclusive
void Network::stochastic_gradient_descent(int startIndex, int endIndex, double eta){
  //assert(batchData.size()>0);
  assert(endIndex > startIndex);
  assert(startIndex >= 0);

  bVector nabla_b = zeroNetworkBiasVector(msizes);
  wVector nabla_w = zeroNetworkWeightVector(msizes);

  //bVector delta_nabla_b;
  //wVector delta_nabla_w;
  std::pair<bVector,wVector> nablaPair;
  for(int n = startIndex; n < endIndex; n++){ // We can parallelize this for loop with some restructuring
    nablaPair = backprop(trainingData_[n].first,trainingData_[n].second);
    //delta_nabla_b = nablaPair.first;
    //delta_nabla_w = nablaPair.second;
    assert(nablaPair.second.size() == nabla_w.size());
    for(int i = 0; i < nablaPair.second.size(); i++){
      assert(nablaPair.second[i].size() == nabla_w[i].size());
      assert(nablaPair.first[i].size() == nabla_b[i].size());
      for(int j = 0; j < nablaPair.second[i].size(); j++){
        nabla_b[i][j] = nabla_b[i][j] + nablaPair.first[i][j];
        for(int k = 0; k < nablaPair.second[i][j].size(); k++){
          nabla_w[i][j][k] = nabla_w[i][j][k] + nablaPair.second[i][j][k];
        }
      }
    }

    //double mu = eta/batchData.size();
    double mu = eta/(endIndex-startIndex);

    //Now we update the weights and biases
    assert(mWeights_.size() == nabla_w.size());
    for(int i = 0; i < mWeights_.size(); i++){
      assert(mWeights_[i].size() == nabla_w[i].size());
      assert(mBiases_[i].size() == nabla_b[i].size());
      for(int j = 0; j < mWeights_[i].size(); j++){
        mBiases_[i][j] = mBiases_[i][j] - (mu*nabla_b[i][j]);
        for(int k = 0; k < mWeights_[i][j].size(); k++){
          mWeights_[i][j][k] = mWeights_[i][j][k] - (mu*nabla_w[i][j][k]);
        }
      }
    }
    // Now the mWeights_ and mBiases_ have been updated applying stochastic
    // gradient descent using backpropagation for the chunk of data contained
    // in batchData.
  }
}

void Network::train_network(int numEpochs, int batchSize, double eta, bool test){
  assert(batchSize > 0);
  bVector imageData = read_MNIST_image_data("../../MNIST_Data/train-images-idx3-ubyte");
  dVector labelData = read_MNIST_label_data("../../MNIST_Data/train-labels-idx1-ubyte");
  //std::vector<std::pair<dVector,dVector> > trainingData = pair_image_label_data(imageData,labelData);
  trainingData_ = pair_image_label_data(imageData,labelData);
  //assert(batchSize <= trainingData.size());
  assert(batchSize <= trainingData_.size());
  //int numBatches = trainingData.size()/batchSize;
  int numBatches = trainingData_.size()/batchSize;
  //assert(numBatches*batchSize-1< trainingData.size());
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
    //cout << "Begin Epoch " << j << endl;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //std::shuffle(trainingData.begin(),trainingData.end(),std::default_random_engine(seed));
    std::shuffle(trainingData_.begin(),trainingData_.end(),std::default_random_engine(seed));
    //cout << "Data shuffled" << endl;
    int start, end;
    for(int i = 0; i < numBatches; i++){
      //cout << "Batch " << i<< "/" << numBatches << endl;
      start = i*batchSize;
      end = (i+1)*batchSize-1;
      if(end > trainingData_.size()){
        end = trainingData_.size();
      }
      //std::vector<std::pair<dVector,dVector> > batchData(trainingData.begin()+start,trainingData.begin()+end);
      //std::vector<std::pair<dVector,dVector> > batchData(trainingData_.begin()+start,trainingData_.begin()+end);
      //stochastic_gradient_descent(batchData, eta);
      stochastic_gradient_descent(start,end,eta);
    }
    if(test){
      std::cout << "Epoch " << j << ": " << test_network(testData) << "/" << testLabelData.size() << std::endl;
    }
    else {
      std::cout << "Epoch " << j << " complete." << std::endl;
    }
  }
}

int Network::test_network(std::vector<std::pair<dVector,dVector> >& testData){
  std::cout << "Start of network testing" << std::endl;
  int numCorrectlyIdentified = 0;
  dVector networkOutput;
  int maxIndex;
  for(int i = 0; i < testData.size(); i++){
    networkOutput = feedforward(testData[i].first);
    maxIndex = find_max_index(networkOutput);
    if(testData[i].second[maxIndex] == 1.0){
      numCorrectlyIdentified++;
    }
    /* For seeing the images that aren't being correctly identified. */
    /*
    else{
      printImage(testData[i].first);
    }*/
  }
  return numCorrectlyIdentified;
};

// Method to save the network biases and weights. Saved as a binary file
void Network::save_network_parameters(std::string filename){
  std::ofstream outfile(filename, std::ofstream::binary | std::ios::out);
  assert(outfile.is_open());

  // First the bias data is saved
  for(int i = 0; i < mBiases_.size(); i++){
    for(int j = 0; j < mWeights_[i].size(); j++){
      outfile.write((char*)&mBiases_[i][j], sizeof(double));
    }
  }

  // Next, save the weights data
  for(int i = 0; i < mWeights_.size(); i++){
    for(int j = 0; j < mWeights_[i].size(); j++){
      for(int k = 0; k < mWeights_[i][j].size(); k++){
        outfile.write((char*)&mWeights_[i][j][k], sizeof(double));
      }
    }
  }

  // Close the file
  outfile.close();
  std::cout << "Saved network parameters to " << filename << std::endl;
};

void Network::load_network_parameters(std::string filename){
  std::ifstream dataFile(filename, std::ios::binary);
  assert(dataFile.is_open());

  double number;

  // First read the bias data
  for(int i = 0; i < mBiases_.size(); i++){
    for(int j = 0; j < mWeights_[i].size(); j++){
      dataFile.read((char*)&number,sizeof(number));
      mBiases_[i][j] = number;
    }
  }

  // Next, read the weights data
  for(int i = 0; i < mWeights_.size(); i++){
    for(int j = 0; j < mWeights_[i].size(); j++){
      for(int k = 0; k < mWeights_[i][j].size(); k++){
        dataFile.read((char*)&number,sizeof(number));
        mWeights_[i][j][k] = number;
      }
    }
  }
  dataFile.close();
  std::cout << "Loaded network parameters from " << filename << std::endl;
};

#endif
