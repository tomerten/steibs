#include <ibs>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <ste>

// PYBIND11_MAKE_OPAQUE(std::vector<double>);
namespace py = pybind11;

PYBIND11_MODULE(STELib, m) {
  m.doc() = "Python wrapper around C++ STELib.";
  /*
   */
  m.def("ran3", &ste_random::ran3, "");
  m.def("BiGaussian4D", &ste_random::BiGaussian4D, "");
  m.def("BiGaussian6D", &ste_random::BiGaussian6D, "");
  m.def("BiGaussian6DLongMatched", &ste_random::BiGaussian6DLongMatched, "");
  m.def("GenerateDistribution", &ste_random::GenerateDistribution, "");
  m.def("GenerateDistributionMatched", &ste_random::GenerateDistributionMatched,
        "");
  m.def("Hamiltonian", &ste_longitudinal::Hamiltonian, "");
  m.def("CalculateEmittance", &ste_global::CalculateEmittance, "");
  m.def("radiationEquilib", &ste_radiation::radiationEquilib, "");
  m.def("RadUpdate", &ste_radiation::RadUpdate, "");
  m.def("updateTwissHeaderLong",
        [](std::map<std::string, double> twissheader,
           std::vector<double> harmonicNumbers, std::vector<double> rfVoltages,
           double aatom, double sigs) {
          ste_longitudinal::updateTwissHeaderLong(twissheader, harmonicNumbers,
                                                  rfVoltages, aatom, sigs);
          return twissheader;
        },
        "");
}