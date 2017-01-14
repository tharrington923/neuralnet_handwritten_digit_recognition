#ifndef READ_CPP_
#define READ_CPP_

#include "aliases.hpp"
#include <string>
#include <cassert>
#include <fstream>
#include <iostream>

int big_to_little_endian(int bigEndianInt){
  unsigned char byte1, byte2, byte3, byte4; // chars used because they are one byte. 32 bit integer needs 4 bytes
  byte1 = bigEndianInt&255; // Bitwise & with 255 selects first 8 bits only (remaining bits are 0). MSB in big endian
  byte2 = (bigEndianInt>>8)&255; // Selects 2nd byte
  byte3 = (bigEndianInt>>16)&255; // Selects 3rd byte.
  byte4 = (bigEndianInt>>24)&255; // Selects 4th byte (LSB in big endian). Will become 1st byte
  return ((int)byte4) + ((int)byte3<<8) + ((int)byte2<<16) + ((int) byte1<<24);
}

bVector read_MNIST_image_data(std::string filename){

  char* buffer;
  int magicNumber;
  int numberImages;
  int numRows;
  int numCols;
  bVector dataOutput;
  dVector singleImage;
  std::ifstream dataFile(filename, std::ios::binary);
  assert(dataFile.is_open());
  dataFile.read((char*)&magicNumber,sizeof(magicNumber));
  magicNumber = big_to_little_endian(magicNumber);
  assert(magicNumber == 2051);
  dataFile.read((char*)&numberImages,sizeof(numberImages));
  numberImages = big_to_little_endian(numberImages);
  // The file is a label file
  //cout << numberImages << endl;
  dataFile.read((char*)&numRows,sizeof(numRows));
  numRows = big_to_little_endian(numRows);
  dataFile.read((char*)&numCols,sizeof(numCols));
  numCols = big_to_little_endian(numCols);
  //cout << numRows << " x " << numCols << endl;
  unsigned char pixel = 0;
  double pixelValue;
  for(int n = 0; n < numberImages; n++){
    singleImage.clear();
    for(int i = 0; i < numRows; i++){
      for(int j = 0; j < numCols; j++){
        dataFile.read((char*)&pixel,sizeof(pixel));
        pixelValue = ((double) pixel)/255.0;
        // Convert to greyscale: White = 0 & Black = 1
        singleImage.push_back(pixelValue);
      }
    }
    dataOutput.push_back(singleImage);
  }
  //cout << dataOutput.size() << endl;
  //cout << dataOutput[0].size() << endl;
  for (int i = 0; i < dataOutput.size(); i++){
    assert(dataOutput[i].size() == 784);
  }
  std::cout << "Loaded: " << filename << std::endl;
  return dataOutput;
};

// Threshold value is a double between 0 and 1. (Below threshold->0, Above -> 1)
bVector read_MNIST_image_data_BW(std::string filename, double threshold){

  char* buffer;
  int magicNumber;
  int numberImages;
  int numRows;
  int numCols;
  bVector dataOutput;
  dVector singleImage;
  std::ifstream dataFile(filename, std::ios::binary);
  assert(dataFile.is_open());
  dataFile.read((char*)&magicNumber,sizeof(magicNumber));
  magicNumber = big_to_little_endian(magicNumber);
  assert(magicNumber == 2051);
  dataFile.read((char*)&numberImages,sizeof(numberImages));
  numberImages = big_to_little_endian(numberImages);
  // The file is a label file
  //cout << numberImages << endl;
  dataFile.read((char*)&numRows,sizeof(numRows));
  numRows = big_to_little_endian(numRows);
  dataFile.read((char*)&numCols,sizeof(numCols));
  numCols = big_to_little_endian(numCols);
  //cout << numRows << " x " << numCols << endl;
  unsigned char pixel = 0;
  double pixelValue;
  for(int n = 0; n < numberImages; n++){
    singleImage.clear();
    for(int i = 0; i < numRows; i++){
      for(int j = 0; j < numCols; j++){
        dataFile.read((char*)&pixel,sizeof(pixel));
        // Convert to greyscale: White = 0 & Black = 1
        pixelValue = ((double) pixel)/255.0;
        // Scale gray values to 0 or 1 by setting threshold
        if(pixelValue < threshold){
          singleImage.push_back(0.0);
        }
        else {
          singleImage.push_back(1.0);
        }
      }
    }
    dataOutput.push_back(singleImage);
  }
  for (int i = 0; i < dataOutput.size(); i++){
    assert(dataOutput[i].size() == 784);
  }
  std::cout << "Loaded: " << filename << std::endl;
  return dataOutput;
};

dVector read_MNIST_label_data(std::string filename){

  char* buffer;
  int magicNumber;
  int numberImages;
  int numRows;
  int numCols;
  dVector dataOutput;
  std::ifstream dataFile(filename, std::ios::binary);
  assert(dataFile.is_open());
  dataFile.read((char*)&magicNumber,sizeof(magicNumber));
  magicNumber = big_to_little_endian(magicNumber);
  assert(magicNumber == 2049);
  dataFile.read((char*)&numberImages,sizeof(numberImages));
  numberImages = big_to_little_endian(numberImages);
  // The file is a label file
  //cout << numberImages << endl;
  unsigned char pixel = 0;
  for(int i = 0; i < numberImages; i++){
    dataFile.read((char*)&pixel,sizeof(pixel));
    dataOutput.push_back((double) pixel);
  }
  assert(dataOutput.size() == numberImages);
  //cout << dataOutput.size() << endl;
  std::cout << "Loaded: " << filename << std::endl;
  return dataOutput;
};

// Note this method assumes that the data is a 28x28 image.
void printImage(dVector& imageVector){
  assert(imageVector.size()==784);
  for(int i = 0; i < 28; i++){
    for (int j = 0; j < 28; j++){
      std::cout << imageVector[i*28+j] << " ";
    }
    std::cout << std::endl;
  }
}

#endif
