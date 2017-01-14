#ifndef NETWORKFUNCTIONS_CPP_
#define NETWORKFUNCTIONS_CPP_

#include "aliases.hpp"
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>

/* These functions are used in the Network class' method definitions */

double sigmoid(double z){
  /* The sigmoid function sigmoid(z) for a single double */
  double out = 1.0/(1.0+std::exp(-1.0*z));
  assert(out >= 0 && out <= 1);
  return out;
}

double sigmoid_prime(double z){
  /* Derivative of the sigmoid function */
  double out = sigmoid(z)*(1-sigmoid(z));
  assert(out >= 0 && out <= 1);
  return out;
}

dVector sigmoid_vector(dVector& inputVector){
  /* Returns a vector with the same length as inputVector where each element
  is sigmoid(z) where z is the original element of inputVector */
  dVector outputVector(inputVector.size());
  for(int i = 0; i < inputVector.size(); i++){
    outputVector[i] = sigmoid(inputVector[i]);
  }
  return outputVector;
}

dVector sigmoid_prime_vector(dVector& inputVector){
  /* Returns a vector with the same length as inputVector where each element
  is sigmoid(z) where z is the original element of inputVector */
  dVector outputVector(inputVector.size());
  for(int i = 0; i < inputVector.size(); i++){
    outputVector[i] = sigmoid_prime(inputVector[i]);
  }
  return outputVector;
}

double vector_dot(dVector& vectorA, dVector& vectorB){
  double out = 0.0;
  assert(vectorA.size() == vectorB.size());
  for(int i = 0; i < vectorA.size(); i++){
    out += (vectorA[i]*vectorB[i]);
  }
  return out;
}

dVector vector_dot(bVector& vectorA, dVector& vectorB){
  assert(vectorA[0].size() == vectorB.size());
  dVector outputVector(vectorA.size());
  for(int i = 0; i < vectorA.size(); i++){
    outputVector[i] = vector_dot(vectorA[i],vectorB);
  }
  return outputVector;
}

dVector vector_add(dVector& vectorA, dVector& vectorB){
  assert(vectorA.size() == vectorB.size());
  dVector outputVector(vectorA.size());
  for(int i = 0; i < vectorA.size(); i++){
    outputVector[i] = (vectorA[i]+vectorB[i]);
  }
  return outputVector;
}

dVector vector_subtract(dVector& vectorA, dVector& vectorB){
  assert(vectorA.size() == vectorB.size());
  dVector outputVector(vectorA.size());
  for(int i = 0; i < vectorA.size(); i++){
    outputVector[i] = (vectorA[i]-vectorB[i]);
  }
  return outputVector;
}

dVector vector_multiply(dVector& vectorA, dVector& vectorB){
  assert(vectorA.size() == vectorB.size());
  dVector outputVector(vectorA.size());
  for(int i = 0; i < vectorA.size(); i++){
    outputVector[i] = (vectorA[i]*vectorB[i]);
  }
  return outputVector;
}

/* This vector product takes a vector vectorA of length n and vector vectorB of
   of length m and produces an nxm matrix */
bVector vector_outer_product(dVector& vectorA, dVector& vectorB){
  bVector outputVector;
  dVector row;
  for(int i = 0; i < vectorA.size(); i++){
    row.clear();
    for(int j = 0; j < vectorB.size(); j++){
      row.push_back(vectorA[i]*vectorB[j]);
    }
    outputVector.push_back(row);
  }
  assert(outputVector.size() == vectorA.size());
  assert(outputVector[0].size() == vectorB.size());
  return outputVector;
}

/* This function take a matrix matrixA of dimensions mxn, transposes it, and
   performs matrix multiplication with a mx1 matrix (vectorA). A vector
  outputVector with dimensions 1xn is produced. */
dVector vector_transpose_product(bVector& matrixA, dVector& vectorB){
  assert(matrixA.size() == vectorB.size());
  dVector outputVector;
  double sum;
  for(int j = 0; j < matrixA[0].size(); j++){
    sum = 0;
    for(int i = 0; i < matrixA.size(); i++){
      sum += matrixA[i][j]*vectorB[i];
    }
    outputVector.push_back(sum);
  }
  assert(outputVector.size() == matrixA[0].size());
  return outputVector;
}

/* This function returns a zero bias matrix that is represented as a vector
   of vectors. The vector contains (# of layers) bias vectors. The bias vector
   for each layer contains (# of neurons in that layer) biases.  */
bVector zeroNetworkBiasVector(std::vector<int> sizes){
  bVector outputVector;

  dVector rowVector;
  // For each of the layers (excluding the first layer),
  for(int i = 0; i < sizes.size()-1; i++){
    rowVector.clear(); // Clear the addBiasVector
    // Create a vector of length equal to the number of neurons in level i+1
    rowVector.resize(sizes[i+1],0.0);
    // Add the layer to output vector
    outputVector.push_back(rowVector);
  }
  return outputVector;
}

/* This function returns a vector containing #a of (b x c) zero matrices
   represented by a vector of (vectors of vectors). This will be used to stored
   the weight matrices for each neuron in every layer. */
wVector zeroNetworkWeightVector(std::vector<int> sizes){
  wVector outputVector;

  // Create a weight vector that will be used
  dVector rowVector;
  bVector addWeightVector;
  // For each of the layers (excluding the first layer),
  for(int i = 0; i < sizes.size()-1; i++){
    addWeightVector.clear(); // Clear the addWeightVector
    // Create a vector of length equal to the number of neurons in that level
    for(int j = 0; j < sizes[i+1]; j++){
      rowVector.clear();
      rowVector.resize(sizes[i],0.0);
      // Push the (#neurons in level j)x(#neurons in level j-1) to the
      addWeightVector.push_back(rowVector);
    }
    // Add the bias vector containing a bias for each neuron in that level to
    // the vector containing the bias vectors for each level
    outputVector.push_back(addWeightVector);
  }

  return outputVector;
}

/* change int_label_to_dVector name and change default length */

// This method ASSSUMES that the output layer has 10 neurons
dVector int_label_to_dVector(double i){
  dVector outputVector(10,0.0);
  outputVector[i] = 1.0;
  assert(outputVector.size() == 10);
  return outputVector;
}

std::vector<std::pair<dVector,dVector> > pair_image_label_data(bVector& imageData, dVector& labelData){
  std::vector<std::pair<dVector,dVector> > outputData;
  assert(imageData.size() == labelData.size());
  std::pair<dVector,dVector> pairData;
  for(int i = 0; i < imageData.size(); i++){
    pairData.first = imageData[i];
    pairData.second = int_label_to_dVector(labelData[i]);
    outputData.push_back(pairData);
  }
  return outputData;
}

int find_max_index(dVector& inputVector){
  double maxValue = -10.0;
  int maxIndex = -1;
  for(int i = 0; i < inputVector.size(); i++){
    //cout << inputVector[i] << endl;
    if(inputVector[i] > maxValue){
      maxIndex = i;
      maxValue = inputVector[i];
    }
  }
  return maxIndex;
}

#endif
