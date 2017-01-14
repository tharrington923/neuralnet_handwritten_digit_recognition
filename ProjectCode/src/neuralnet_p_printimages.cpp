#include "network.hpp"
#include "network.cpp"
#include "pnetwork.hpp"
#include "pnetwork.cpp"
#include <vector>
#include <cassert>
#include <cstdlib>


int main(int argc, const char * argv[]) {

  int numImag = 10;
  if (argc > 1) {
    std::cout << "Printing " << argv[1] << " images." << std::endl;
    numImag = std::atoi(argv[1]);
  }



  bVector testImageData;

  testImageData = read_MNIST_image_data_BW("../../MNIST_Data/t10k-images-idx3-ubyte",0.7);
  for(int i = 0; i < numImag; i++){
    printImage(testImageData[i]);
    std::cout << "________________________________________________________" << std::endl;
  }


  return 0;
}
