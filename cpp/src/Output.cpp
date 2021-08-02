#include "../include/ste_bits/Output.hpp"
#include <algorithm>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <vector>

namespace ste_output {
void red() { printf("\033[1;31m"); }
void yellow() { printf("\033[1;33m"); }
void green() { printf("\033[1;32m"); }
void blue() { printf("\033[1;34m"); }
void cyan() { printf("\033[1;36m"); }
void reset() { printf("\033[0m"); }

void printVector(std::vector<double> vec) {
  std::copy(vec.begin(), vec.end(),
            std::ostream_iterator<double>(std::cout, " "));
  std::cout << std::endl;
}
} // namespace ste_output
