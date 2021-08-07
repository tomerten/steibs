#ifndef RADIATION_H
#define RADIATION_H
// #include <CL/cl.h>
#include <map>
#include <string>
#include <vector>

namespace ste_radiation {

std::map<std::string, double>
radiationIntegrals(std::map<std::string, std::vector<double>> &twiss);

std::map<std::string, double>
radiationEquilib(std::map<std::string, double> &twiss);

double RadiationLossesPerTurn(std::map<std::string, double> &twiss);

void RadUpdate(std::vector<std::vector<double>> &distribution,
               std::map<std::string, double> &tw,
               std::map<std::string, double> &radparam, int &seed);
} // namespace ste_radiation
#endif
