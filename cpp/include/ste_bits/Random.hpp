#ifndef RANDOM_H
#define RANDOM_H
// #include <CL/cl.h>
#include <map>
#include <string>
#include <vector>

namespace ste_random {

double ran3(int *idum);
std::vector<double> BiGaussian4D(double betax, double ex, double betay,
                                 double ey, int seed);

std::vector<double> BiGaussian6D(double betax, double ex, double betay,
                                 double ey,
                                 std::vector<double> &harmonicNumbers,
                                 std::vector<double> &rfVoltages,
                                 std::map<std::string, double> &twissheadermapL,
                                 int seed);

std::vector<double> BiGaussian6DLongMatched(
    double betax, double ex, double betay, double ey,
    std::vector<double> &harmonicNumbers, std::vector<double> &rfVoltages,
    std::map<std::string, double> &twissheadermapL, int seed);

std::vector<std::vector<double>>
GenerateDistribution(int nMacro, double betax, double ex, double betay,
                     double ey, std::vector<double> &harmonicNumbers,
                     std::vector<double> &rfVoltages,
                     std::map<std::string, double> &twissheadermapL, int seed);

std::vector<std::vector<double>> GenerateDistributionMatched(
    int nMacro, double betax, double ex, double betay, double ey,
    std::vector<double> &harmonicNumbers, std::vector<double> &rfVoltages,
    std::map<std::string, double> &twissheadermapL, int seed);
} // namespace ste_random

#endif