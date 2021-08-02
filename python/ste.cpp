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
}