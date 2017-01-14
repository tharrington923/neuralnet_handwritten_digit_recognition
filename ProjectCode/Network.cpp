// HelloWorld.cpp
#include <iostream>
#include <math.h>
#include <cmath>
#include <random>
#include <cassert>
#include <fstream>
#include <string>
#include <algorithm>

using namespace std;

typedef vector<double> dVector;
typedef vector<vector<double> > bVector;
typedef vector<vector<vector<double> > > wVector;

/*******************************************************************************
                          Miscellaneous Functions
*******************************************************************************/

double sigmoid(double z){
  /* The sigmoid function sigmoid(z) for a single double */
  double out = 1.0/(1.0+exp(-1.0*z));
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
bVector zeroNetworkBiasVector(vector<int> sizes){
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
wVector zeroNetworkWeightVector(vector<int> sizes){
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

int big_to_little_endian(int bigEndianInt){
  unsigned char byte1, byte2, byte3, byte4; // chars used because they are one byte. 32 bit integer needs 4 bytes
  byte1 = bigEndianInt&255; // Bitwise & with 255 selects first 8 bits only (remaining bits are 0). MSB in big endian
  byte2 = (bigEndianInt>>8)&255; // Selects 2nd byte
  byte3 = (bigEndianInt>>16)&255; // Selects 3rd byte.
  byte4 = (bigEndianInt>>24)&255; // Selects 4th byte (LSB in big endian). Will become 1st byte
  return ((int)byte4) + ((int)byte3<<8) + ((int)byte2<<16) + ((int) byte1<<24);
}

bVector read_MNIST_image_data(string filename){

  char* buffer;
  int magicNumber;
  int numberImages;
  int numRows;
  int numCols;
  bVector dataOutput;
  dVector singleImage;
  ifstream dataFile(filename, ios::binary);
  assert(dataFile.is_open());
  dataFile.read((char*)&magicNumber,sizeof(magicNumber));
  magicNumber = big_to_little_endian(magicNumber);
  assert(magicNumber == 2051);
  dataFile.read((char*)&numberImages,sizeof(numberImages));
  numberImages = big_to_little_endian(numberImages);
  // The file is a label file
  cout << numberImages << endl;
  dataFile.read((char*)&numRows,sizeof(numRows));
  numRows = big_to_little_endian(numRows);
  dataFile.read((char*)&numCols,sizeof(numCols));
  numCols = big_to_little_endian(numCols);
  cout << numRows << " x " << numCols << endl;
  unsigned char pixel = 0;
  for(int n = 0; n < numberImages; n++){
    singleImage.clear();
    for(int i = 0; i < numRows; i++){
      for(int j = 0; j < numCols; j++){
        dataFile.read((char*)&pixel,sizeof(pixel));
        // Convert to greyscale: White = 0 & Black = 1
        singleImage.push_back(((double) pixel)/255.0);
      }
    }
    dataOutput.push_back(singleImage);
  }
  cout << dataOutput.size() << endl;
  cout << dataOutput[0].size() << endl;
  for (int i = 0; i < dataOutput.size(); i++){
    assert(dataOutput[i].size() == 784);
  }
  return dataOutput;
};

dVector read_MNIST_label_data(string filename){

  char* buffer;
  int magicNumber;
  int numberImages;
  int numRows;
  int numCols;
  dVector dataOutput;
  ifstream dataFile(filename, ios::binary);
  assert(dataFile.is_open());
  dataFile.read((char*)&magicNumber,sizeof(magicNumber));
  magicNumber = big_to_little_endian(magicNumber);
  assert(magicNumber == 2049);
  dataFile.read((char*)&numberImages,sizeof(numberImages));
  numberImages = big_to_little_endian(numberImages);
  // The file is a label file
  cout << numberImages << endl;
  unsigned char pixel = 0;
  for(int i = 0; i < numberImages; i++){
    dataFile.read((char*)&pixel,sizeof(pixel));
    dataOutput.push_back((double) pixel);
  }
  assert(dataOutput.size() == numberImages);
  cout << dataOutput.size() << endl;
  return dataOutput;
};

// This method ASSSUMES that the output layer has 10 neurons
dVector int_label_to_dVector(double i){
  dVector outputVector(10,0.0);
  outputVector[i] = 1.0;
  assert(outputVector.size() == 10);
  return outputVector;
}

vector<pair<dVector,dVector> > pair_image_label_data(bVector imageData, dVector labelData){
  vector<pair<dVector,dVector> > outputData;
  assert(imageData.size() == labelData.size());
  pair<dVector,dVector> pairData;
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


/*******************************************************************************
                      End of Miscellaneous Functions
*******************************************************************************/

class Network {
  public:
//private:
  int num_layers; // Number of layers in neural Network
  vector<int> msizes; // Number of neurons in each layer

  bVector mBiases_; // Neuron biases. Vector of Vectors. One for each neuron in each layer (excluding input)
  wVector mWeights_; // Neuron weights. Vector of Matrices (represented as Vector of Vectors)

//public:

  /* The class constructor that takes a vector<int> sizes
     sizes contains the number of neurons in each layer of the neural network.
     The first layer is assumed to be an input layer, so there are no biases for
     the input layer. */
  Network(vector<int> sizes) {
    msizes = sizes;
    num_layers = msizes.size();
    int seed = 1;
    // Knuth_b random number generator
    knuth_b generator(seed);
    // Normal distribution with mean 0.0 and a std dev 1.0
    normal_distribution<double> distribution(0.0,1.0);

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

  Network(vector<int> sizes, bVector& biases, wVector& weights) {
    msizes = sizes;
    num_layers = msizes.size();
    int seed = 1;
    // Knuth_b random number generator
    knuth_b generator(seed);
    // Normal distribution with mean 0.0 and a std dev 1.0
    normal_distribution<double> distribution(0.0,1.0);

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

  dVector feedforward(dVector& inputVector){
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
  pair<bVector, wVector > backprop(dVector& inputVector, dVector& expectedOutput){

    pair<bVector, wVector > nablaPair;
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

  dVector cost_derivative(dVector& output_activations, dVector& y){
    return vector_subtract(output_activations,y);
  }

  // batchData is the set of x and y stored in a vector of pairs
  void stochastic_gradient_descent(vector<pair<dVector,dVector> >& batchData, double eta){
    assert(batchData.size()>0);

    bVector nabla_b = zeroNetworkBiasVector(msizes);
    wVector nabla_w = zeroNetworkWeightVector(msizes);

    bVector delta_nabla_b;
    wVector delta_nabla_w;
    pair<bVector,wVector> nablaPair;
    for(int n = 0; n < batchData.size(); n++){ // We can parallelize this for loop with some restructuring
      nablaPair = backprop(batchData[n].first,batchData[n].second);
      delta_nabla_b = nablaPair.first;
      delta_nabla_w = nablaPair.second;
      assert(delta_nabla_w.size() == nabla_w.size());
      for(int i = 0; i < delta_nabla_w.size(); i++){
        assert(delta_nabla_w[i].size() == nabla_w[i].size());
        assert(delta_nabla_b[i].size() == nabla_b[i].size());
        for(int j = 0; j < delta_nabla_w[i].size(); j++){
          nabla_b[i][j] = nabla_b[i][j] + delta_nabla_b[i][j];
          for(int k = 0; k < delta_nabla_w[i][j].size(); k++){
            nabla_w[i][j][k] = nabla_w[i][j][k] + delta_nabla_w[i][j][k];
          }
        }
      }

      double mu = eta/batchData.size();

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

  void train_network(int numEpochs, int batchSize, double eta, bool test){
    assert(batchSize > 0);
    bVector imageData = read_MNIST_image_data("../MNIST_Data/train-images-idx3-ubyte");
    dVector labelData = read_MNIST_label_data("../MNIST_Data/train-labels-idx1-ubyte");
    vector<pair<dVector,dVector> > trainingData = pair_image_label_data(imageData,labelData);
    assert(batchSize <= trainingData.size());
    int numBatches = trainingData.size()/batchSize;
    assert(numBatches*batchSize-1< trainingData.size());
    bVector testImageData;
    dVector testLabelData;
    vector<pair<dVector,dVector> > testData;
    if(test){
      testImageData = read_MNIST_image_data("../MNIST_Data/t10k-images-idx3-ubyte");
      testLabelData = read_MNIST_label_data("../MNIST_Data/t10k-labels-idx1-ubyte");
      testData = pair_image_label_data(testImageData,testLabelData);
    }
    for(int j = 0; j < numEpochs; j++){
      // Shuffle trainingData
      //cout << "Begin Epoch " << j << endl;
      random_shuffle(trainingData.begin(),trainingData.end());
      //cout << "Data shuffled" << endl;
      int start, end;
      for(int i = 0; i < numBatches; i++){
        //cout << "Batch " << i<< "/" << numBatches << endl;
        start = i*batchSize;
        end = (i+1)*batchSize-1;
        vector<pair<dVector,dVector> > batchData(trainingData.begin()+start,trainingData.begin()+end);
        stochastic_gradient_descent(batchData, eta);
      }
      if(test){
        cout << "Epoch " << j << ": " << test_network(testData) << "/" << testLabelData.size() << endl;
      }
      else {
        cout << "Epoch " << j << " complete." << endl;
      }
    }
  }

  int test_network(vector<pair<dVector,dVector> >& testData){
    int numCorrectlyIdentified = 0;
    dVector networkOutput;
    int maxIndex;
    for(int i = 0; i < testData.size(); i++){
      networkOutput = feedforward(testData[i].first);
      maxIndex = find_max_index(networkOutput);
      if(testData[i].second[maxIndex] == 1.0){
        numCorrectlyIdentified++;
      }
      /*
      if(abs(testData[i].second[maxIndex] - 1.0)<0.000001){
        numCorrectlyIdentified++;
      }*/
    }
    return numCorrectlyIdentified;
  };

  // Method to save the network biases and weights. Saved as a binary file
  void save_network_parameters(string filename){
    std::ofstream outfile(filename, ofstream::binary | ios::out);
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
  };

  void load_network_parameters(string filename){
    ifstream dataFile(filename, ios::binary);
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
  };

};



void test_network_methods(){
  cout << "Testing network" << endl;

  vector<int> a = {2,3,1};

  bVector biases = {{0.30926276,-2.01485405,1.01445334},{1.35023938}};

  wVector weights = {{{-0.79845271,0.73383224},{-0.47689875,0.04373985},{-1.08138889,-1.09604203}},{{-1.3281524,-2.36424472,-0.30025259}}};

  Network b(a,biases,weights);

  dVector ff = {2.0,5.0};
  dVector ffout;
  cout << "Testing feedforward" << endl;
  ffout = b.feedforward(ff);
  assert(abs(ffout[0] - 0.49799423) <= 0.000001);
  cout << "feedforward successful" << endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  cout << "Testing backpropagation" << endl;
  pair<bVector, wVector > nablaPair;
  nablaPair = b.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{2.58147444e-02, 6.45368611e-02},{3.35156908e-02, 8.37892271e-02},{9.93744837e-05, 2.48436209e-04}},{{-0.11488306, -0.00754118, -0.0001657}}};
  bVector nabla_b_Values = {{1.29073722e-02,1.67578454e-02, 4.96872418e-05},{-0.12549942}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.00001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.00001);
      }
    }
  }
  cout << "Backpropagation algorithm successful" << endl;
};

void test_network_methods2(){
  cout << "Testing network" << endl;

  vector<int> a = {2,3,4,5,1};

  bVector biases = {{-1.48605702,-0.71319336,-0.49098387}, { 0.99883046,-0.90136997,-0.9994927,0.36354862}, {0.29321565, 0.27842404, 1.42378113, 0.96487395, 1.72427438}, {1.73059788}};

  wVector weights = {{{0.90379991, -1.50240658},
                                              {0.62670131,0.4565706},
                                              {-0.30867027,-1.08305284}},
                                             {{0.44174655,-1.3246689,0.83697628},
                                              {-2.94722175,-1.14075892,-0.03972066},
                                              {-0.24397608,0.35861847,0.35607926},
                                              {0.34916269, -0.36702888,0.05093519}},
                                             {{0.9699126,1.42612543,-0.10184366,0.29009277},
                                              {1.94110995,0.0988707,-1.47877964, -1.01992391},
                                              {0.14510384,-0.03072929,-0.29882381,-0.19205623},
                                              {1.33289393,-0.53430868,2.03604052,-1.17521863},
                                              {0.59632638,-0.72107636,0.89496499,-0.55038029}},
                                             {{0.6660476,-0.69964228, -1.3822408 ,0.9402472 ,  0.14463772}}};

  Network b(a,biases,weights);

  dVector ff = {2.0,5.0};
  dVector ffout;
  cout << "Testing feedforward" << endl;
  ffout = b.feedforward(ff);
  assert(abs(ffout[0] - 0.84163425) <= 0.000001);
  cout << "feedforward successful" << endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  cout << "Testing backpropagation" << endl;
  pair<bVector, wVector > nablaPair;
  nablaPair = b.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{1.91862764e-06,4.79656911e-06},{-1.05574923e-04,-2.63937306e-04},{-2.26300352e-06,-5.65750881e-06}},{{2.30364091e-07,2.88710447e-04,4.48473112e-07},{-1.59240419e-07,-1.99572652e-04,-3.10009455e-07},{-2.13940026e-06,-2.68126513e-03,-4.16498720e-06},{-3.80925501e-07,-4.77405880e-04,-7.41586260e-07}},{{-1.20885427e-03,-3.34839640e-04,-9.40384581e-04,-1.39254982e-03},{1.61044492e-03,4.46075935e-04,1.25278755e-03,1.85516555e-03},{2.16838314e-03,6.00618825e-04,1.68681546e-03,2.49788716e-03},{-1.22708888e-03,-3.39890431e-04,-9.54569539e-04,-1.41355534e-03},{-1.48488701e-04,-4.11297744e-05,-1.15511430e-04,-1.71052806e-04}},{{-0.01544129,-0.01119811,-0.01653364,-0.01751589,-0.01842113}}};
  bVector nabla_b_Values = {{9.59313821e-07,-5.27874613e-05,-1.13150176e-06}, {0.00030587,-0.00021143,-0.0028406,-0.00050578}, {-0.00276102,0.00367825,0.00495258,-0.00280267,-0.00033915}, {-0.02110794}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.0000001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.0000001);
      }
    }
  }
  cout << "Backpropagation algorithm successful" << endl;

  vector<pair<dVector,dVector> > batchData;
  dVector bdVec = {3.0,4.0};
  dVector oVec = {5.4};
  pair<dVector,dVector> tPair(bdVec,oVec);
  double eta = 3.0;
  batchData.push_back(tPair);
  b.stochastic_gradient_descent(batchData, eta);
};

void test_find_max_index(){
  dVector testLabelData = read_MNIST_label_data("../MNIST_Data/t10k-labels-idx1-ubyte");
  dVector labelVector;
  int maxIndex;
  for(int i = 0; i < testLabelData.size(); i++){
    labelVector = int_label_to_dVector(testLabelData[i]);
    maxIndex = find_max_index(labelVector);
    assert(labelVector[maxIndex] == 1.0);
  }
  cout << "find_max_index algorithm passed testing" << endl;
};

void test_save_load_network_methods(){
  cout << "Testing network" << endl;

  vector<int> a = {2,3,4,5,1};

  bVector biases = {{-1.48605702,-0.71319336,-0.49098387}, { 0.99883046,-0.90136997,-0.9994927,0.36354862}, {0.29321565, 0.27842404, 1.42378113, 0.96487395, 1.72427438}, {1.73059788}};

  wVector weights = {{{0.90379991, -1.50240658},
                                              {0.62670131,0.4565706},
                                              {-0.30867027,-1.08305284}},
                                             {{0.44174655,-1.3246689,0.83697628},
                                              {-2.94722175,-1.14075892,-0.03972066},
                                              {-0.24397608,0.35861847,0.35607926},
                                              {0.34916269, -0.36702888,0.05093519}},
                                             {{0.9699126,1.42612543,-0.10184366,0.29009277},
                                              {1.94110995,0.0988707,-1.47877964, -1.01992391},
                                              {0.14510384,-0.03072929,-0.29882381,-0.19205623},
                                              {1.33289393,-0.53430868,2.03604052,-1.17521863},
                                              {0.59632638,-0.72107636,0.89496499,-0.55038029}},
                                             {{0.6660476,-0.69964228, -1.3822408 ,0.9402472 ,  0.14463772}}};

  Network b(a,biases,weights);

  b.save_network_parameters("testNetworkA");

  Network c(a);

  c.load_network_parameters("testNetworkA");

  cout << c.mBiases_[0][0] << endl;
  cout << c.mWeights_[0][0][0] << endl;

  dVector ff = {2.0,5.0};
  dVector ffout;
  cout << "Testing feedforward" << endl;
  ffout = c.feedforward(ff);
  cout << ffout[0] << endl;
  assert(abs(ffout[0] - 0.84163425) <= 0.000001);
  cout << "feedforward successful" << endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  cout << "Testing backpropagation" << endl;
  pair<bVector, wVector > nablaPair;
  nablaPair = c.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{1.91862764e-06,4.79656911e-06},{-1.05574923e-04,-2.63937306e-04},{-2.26300352e-06,-5.65750881e-06}},{{2.30364091e-07,2.88710447e-04,4.48473112e-07},{-1.59240419e-07,-1.99572652e-04,-3.10009455e-07},{-2.13940026e-06,-2.68126513e-03,-4.16498720e-06},{-3.80925501e-07,-4.77405880e-04,-7.41586260e-07}},{{-1.20885427e-03,-3.34839640e-04,-9.40384581e-04,-1.39254982e-03},{1.61044492e-03,4.46075935e-04,1.25278755e-03,1.85516555e-03},{2.16838314e-03,6.00618825e-04,1.68681546e-03,2.49788716e-03},{-1.22708888e-03,-3.39890431e-04,-9.54569539e-04,-1.41355534e-03},{-1.48488701e-04,-4.11297744e-05,-1.15511430e-04,-1.71052806e-04}},{{-0.01544129,-0.01119811,-0.01653364,-0.01751589,-0.01842113}}};
  bVector nabla_b_Values = {{9.59313821e-07,-5.27874613e-05,-1.13150176e-06}, {0.00030587,-0.00021143,-0.0028406,-0.00050578}, {-0.00276102,0.00367825,0.00495258,-0.00280267,-0.00033915}, {-0.02110794}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.0000001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.0000001);
      }
    }
  }
  cout << "Backpropagation algorithm successful" << endl;
};



int main(int argc, const char * argv[]) {

  vector<int> a = {2,3,1} ; // =  (2,3,1); // = [2,3,1];
  Network b(a);

  //test_feedforward();

  test_network_methods();

  cout << "Test 2" << endl;
  test_network_methods2();

  cout << "Test save_load_network methods" << endl;
  test_save_load_network_methods();

  test_find_max_index();

  read_MNIST_image_data("../MNIST_Data/train-images-idx3-ubyte");

  read_MNIST_label_data("../MNIST_Data/t10k-labels-idx1-ubyte");

  vector<int> c = {784,30,10};
  Network d(c);
  //d.train_network(30, 10, 3.0, true);

  return 0;
}
