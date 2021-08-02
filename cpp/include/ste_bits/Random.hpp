#ifndef RANDOM_H
#define RANDOM_H
// #include <CL/cl.h>
#include <vector>

namespace ste_random {

double ran3(int *idum);
std::vector<double> BiGaussian4D(double betax, double ex, double betay,
                                 double ey, int seed);
// std::vector<double> BiGaussian6D();

} // namespace ste_random

#endif