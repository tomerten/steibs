#ifndef LONGITUDINAL_H
#define LONGITUDINAL_H
// #include <CL/cl.h>
#include <map>
#include <string>
#include <vector>

namespace ste_longitudinal {
void updateTwissHeaderLong(std::map<std::string, double> &twissheader,
                           std::vector<double> &harmonicNumbers,
                           std::vector<double> &rfVoltages, double aatom,
                           double sigs);

double tcoeff(std::map<std::string, double> &twissheaderL,
              double baseHarmonicNumber);

double pcoeff(std::map<std::string, double> &twissheaderL, double voltage);

double Hamiltonian(std::map<std::string, double> &twissheaderL,
                   std::vector<double> &harmonicNumbers,
                   std::vector<double> &rfVoltages, double tcoeff, double t,
                   double delta);

} // namespace ste_longitudinal
#endif