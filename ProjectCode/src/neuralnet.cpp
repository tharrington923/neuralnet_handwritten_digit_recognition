#include "aliases.hpp"
#include "network.hpp"
#include "network.cpp"
#include <vector>
#include <cassert>
#include <fstream>
#include <iostream>
#include <random>
//#include "testfunctions.cpp"

int main(int argc, const char * argv[]) {

  std::vector<int> a = {784,30,10};
  Network b(a);
  b.train_network(30, 10, 3.0, true);
  b.save_network_parameters("trained_network_hiddenlayer1_neurons30_batchsize10_eta3");

  return 0;
}
