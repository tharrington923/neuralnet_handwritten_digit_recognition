#ifndef TESTFUNCTIONS_CPP_
#define TESTFUNCTIONS_CPP_

#include "aliases.hpp"
#include "network.hpp"
#include "networkfunctions.cpp"
#include "network.cpp"
#include "read.cpp"
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>

void test_network_methods(){
  std::cout << "Testing network" << std::endl;

  std::vector<int> a = {2,3,1};

  bVector biases = {{0.30926276,-2.01485405,1.01445334},{1.35023938}};

  wVector weights = {{{-0.79845271,0.73383224},{-0.47689875,0.04373985},{-1.08138889,-1.09604203}},{{-1.3281524,-2.36424472,-0.30025259}}};

  Network b(a,biases,weights);

  dVector ff = {2.0,5.0};
  dVector ffout;
  std::cout << "Testing feedforward" << std::endl;
  ffout = b.feedforward(ff);
  assert(std::abs(ffout[0] - 0.49799423) <= 0.000001);
  std::cout << "feedforward successful" << std::endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  std::cout << "Testing backpropagation" << std::endl;
  std::pair<bVector, wVector > nablaPair;
  nablaPair = b.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{2.58147444e-02, 6.45368611e-02},{3.35156908e-02, 8.37892271e-02},{9.93744837e-05, 2.48436209e-04}},{{-0.11488306, -0.00754118, -0.0001657}}};
  bVector nabla_b_Values = {{1.29073722e-02,1.67578454e-02, 4.96872418e-05},{-0.12549942}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(std::abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.00001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(std::abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.00001);
      }
    }
  }
  std::cout << "Backpropagation algorithm successful" << std::endl;
};

void test_network_methods2(){
  std::cout << "Testing network" << std::endl;

  std::vector<int> a = {2,3,4,5,1};

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
  std::cout << "Testing feedforward" << std::endl;
  ffout = b.feedforward(ff);
  assert(std::abs(ffout[0] - 0.84163425) <= 0.000001);
  std::cout << "feedforward successful" << std::endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  std::cout << "Testing backpropagation" << std::endl;
  std::pair<bVector, wVector > nablaPair;
  nablaPair = b.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{1.91862764e-06,4.79656911e-06},{-1.05574923e-04,-2.63937306e-04},{-2.26300352e-06,-5.65750881e-06}},{{2.30364091e-07,2.88710447e-04,4.48473112e-07},{-1.59240419e-07,-1.99572652e-04,-3.10009455e-07},{-2.13940026e-06,-2.68126513e-03,-4.16498720e-06},{-3.80925501e-07,-4.77405880e-04,-7.41586260e-07}},{{-1.20885427e-03,-3.34839640e-04,-9.40384581e-04,-1.39254982e-03},{1.61044492e-03,4.46075935e-04,1.25278755e-03,1.85516555e-03},{2.16838314e-03,6.00618825e-04,1.68681546e-03,2.49788716e-03},{-1.22708888e-03,-3.39890431e-04,-9.54569539e-04,-1.41355534e-03},{-1.48488701e-04,-4.11297744e-05,-1.15511430e-04,-1.71052806e-04}},{{-0.01544129,-0.01119811,-0.01653364,-0.01751589,-0.01842113}}};
  bVector nabla_b_Values = {{9.59313821e-07,-5.27874613e-05,-1.13150176e-06}, {0.00030587,-0.00021143,-0.0028406,-0.00050578}, {-0.00276102,0.00367825,0.00495258,-0.00280267,-0.00033915}, {-0.02110794}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(std::abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.0000001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(std::abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.0000001);
      }
    }
  }
  std::cout << "Backpropagation algorithm successful" << std::endl;

  //std::vector<std::pair<dVector,dVector> > batchData;
  dVector bdVec = {3.0,4.0};
  dVector oVec = {5.4};
  std::pair<dVector,dVector> tPair(bdVec,oVec);
  double eta = 3.0;
  //batchData.push_back(tPair);
  b.trainingData_.push_back(tPair);
  //b.stochastic_gradient_descent(batchData, eta);
  b.stochastic_gradient_descent(0,1, eta);
};

void test_find_max_index(){
  dVector testLabelData = read_MNIST_label_data("../../MNIST_Data/t10k-labels-idx1-ubyte");
  dVector labelVector;
  int maxIndex;
  for(int i = 0; i < testLabelData.size(); i++){
    labelVector = int_label_to_dVector(testLabelData[i]);
    maxIndex = find_max_index(labelVector);
    assert(labelVector[maxIndex] == 1.0);
  }
  std::cout << "find_max_index algorithm passed testing" << std::endl;
};

void test_save_load_network_methods(){
  std::cout << "Testing network" << std::endl;

  std::vector<int> a = {2,3,4,5,1};

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

  std::cout << c.mBiases_[0][0] << std::endl;
  std::cout << c.mWeights_[0][0][0] << std::endl;

  dVector ff = {2.0,5.0};
  dVector ffout;
  std::cout << "Testing feedforward" << std::endl;
  ffout = c.feedforward(ff);
  std::cout << ffout[0] << std::endl;
  assert(std::abs(ffout[0] - 0.84163425) <= 0.000001);
  std::cout << "feedforward successful" << std::endl;

  dVector in = {2.0,5.0};
  dVector out = {1.0};

  std::cout << "Testing backpropagation" << std::endl;
  std::pair<bVector, wVector > nablaPair;
  nablaPair = c.backprop(in, out);

  // Here the expected results are stored in vectors to check the bp algorithm
  wVector nabla_w_Values = {{{1.91862764e-06,4.79656911e-06},{-1.05574923e-04,-2.63937306e-04},{-2.26300352e-06,-5.65750881e-06}},{{2.30364091e-07,2.88710447e-04,4.48473112e-07},{-1.59240419e-07,-1.99572652e-04,-3.10009455e-07},{-2.13940026e-06,-2.68126513e-03,-4.16498720e-06},{-3.80925501e-07,-4.77405880e-04,-7.41586260e-07}},{{-1.20885427e-03,-3.34839640e-04,-9.40384581e-04,-1.39254982e-03},{1.61044492e-03,4.46075935e-04,1.25278755e-03,1.85516555e-03},{2.16838314e-03,6.00618825e-04,1.68681546e-03,2.49788716e-03},{-1.22708888e-03,-3.39890431e-04,-9.54569539e-04,-1.41355534e-03},{-1.48488701e-04,-4.11297744e-05,-1.15511430e-04,-1.71052806e-04}},{{-0.01544129,-0.01119811,-0.01653364,-0.01751589,-0.01842113}}};
  bVector nabla_b_Values = {{9.59313821e-07,-5.27874613e-05,-1.13150176e-06}, {0.00030587,-0.00021143,-0.0028406,-0.00050578}, {-0.00276102,0.00367825,0.00495258,-0.00280267,-0.00033915}, {-0.02110794}};

  for(int i = 0; i < nabla_w_Values.size(); i++){
    for(int j = 0; j < nabla_w_Values[i].size(); j++){
      assert(std::abs(nabla_b_Values[i][j]-nablaPair.first[i][j])<0.0000001);
      for(int k = 0; k < nabla_w_Values[i][j].size(); k++){
        assert(std::abs(nabla_w_Values[i][j][k]-nablaPair.second[i][j][k])<0.0000001);
      }
    }
  }
  std::cout << "Backpropagation algorithm successful" << std::endl;
};

int main(int argc, const char * argv[]) {

  test_network_methods();

  std::cout << "Test 2" << std::endl;
  test_network_methods2();

  std::cout << "Test save_load_network methods" << std::endl;
  test_save_load_network_methods();

  std::cout << "Testing find_max_index algorithm" << std::endl;
  test_find_max_index();

  std::cout << "Testing read MNIST image data" << std::endl;
  read_MNIST_image_data("../../MNIST_Data/train-images-idx3-ubyte");

  std::cout << "Testing read MNIST label data" << std::endl;
  read_MNIST_label_data("../../MNIST_Data/t10k-labels-idx1-ubyte");

  return 0;
}

#endif
