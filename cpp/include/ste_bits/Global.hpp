#ifndef GLOBAL_H
#define GLOBAL_H
// #include <CL/cl.h>
#include <map>
#include <string>
#include <vector>

namespace ste_global {

std::vector<double> CalculateEmittance(std::vector<std::vector<double>> arr,
                                       double betx, double bety);
} // namespace ste_global
#endif