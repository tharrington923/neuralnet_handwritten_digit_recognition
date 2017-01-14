#include "network.hpp"
#include "network.cpp"
#include "pnetwork.hpp"
#include "pnetwork.cpp"
#include <vector>
#include <cassert>
#include <ctime>
#include <cstdlib>


int main(int argc, const char * argv[]) {

  int numEpochs = 30;
  if (argc > 1) {
    std::cout << "Training for " << argv[1] << " epochs." << std::endl;
    numEpochs = std::atoi(argv[1]);
  }

  int nBatch = 10;
  if (argc > 2) {
    std::cout << "Using " << argv[2] << " images in a batch." << std::endl;
    nBatch = std::atoi(argv[2]);
  }

  std::vector<int> a = {784,30,10};
  pNetwork b(a);

  //b.load_network_parameters("trained_network_hiddenlayer1_neurons30_batchsize10_eta3_p_30_BW");

  bVector testImageData;
  dVector testLabelData;
  std::vector<std::pair<dVector,dVector> > testData;

  std::cout << "Number of threads = " << std::thread::hardware_concurrency() << std::endl;
  std::clock_t start;
  double duration;
  start = std::clock();
  b.train_network(numEpochs, nBatch, 3.0, false, std::thread::hardware_concurrency());
  duration = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
  std::cout << "Duration is " << duration << " seconds." << std::endl;
  b.save_network_parameters("trained_network_parameters");

  testImageData = read_MNIST_image_data_BW("../../MNIST_Data/t10k-images-idx3-ubyte",0.7);
  testLabelData = read_MNIST_label_data("../../MNIST_Data/t10k-labels-idx1-ubyte");
  testData = pair_image_label_data(testImageData,testLabelData);
  std::cout << b.test_network(testData) << "/" << testLabelData.size() << std::endl;

  return 0;
}
