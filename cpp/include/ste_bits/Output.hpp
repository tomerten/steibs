#ifndef OUTPUT_H
#define OUTPUT_H
#include <iostream>
#include <iterator>
#include <vector>

namespace ste_output {
void red();
void yellow();
void green();
void blue();
void cyan();
void reset();

void printVector(std::vector<double> vec);

} // namespace ste_output

#endif