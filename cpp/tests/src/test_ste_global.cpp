#include <ibs>
#include <iomanip>
#include <iostream>
#include <ste>

// for testing
template <typename T>
bool RequireVectorEquals(const std::vector<T> &a, const std::vector<T> &b,
                         T max_error = 0.000000001) {
  const auto first_mismatch = std::mismatch(
      a.begin(), a.end(), b.begin(), [max_error](T val_a, T val_b) {
        // std::cout << val_a << " " << val_b << std::endl;
        return val_a == val_b || std::abs(val_a - val_b) < max_error;
      });
  if (first_mismatch.first != a.end()) {
    // std::printf("%s", std::to_string(*first_mismatch.first).c_str());
    // std::printf("%s", std::to_string(*first_mismatch.second).c_str());
    return false;
  } else
    return true;
}

bool test_calc_emit(std::vector<std::vector<double>> distribution, double betx,
                    double bety) {
  std::vector<double> expected = {5.10117e-09, 1.04194e-08, 2.52591e-10,
                                  1.42116e-09, 1.66734e-11, 0.00114571};

  std::vector<double> actual =
      ste_global::CalculateEmittance(distribution, betx, bety);
  return RequireVectorEquals(expected, actual, 0.1);
}

int main() {
  // set seed
  int seed = 123456;

  string twissfilename = "../src/b2_design_lattice_1996.twiss";
  map<string, double> twheader;
  twheader = GetTwissHeader(twissfilename);
  // rf settings
  std::vector<double> h, v;
  h.push_back(400.0);
  v.push_back(-1.5e6);

  // bunch length
  double sigs = 0.005;

  // aatom
  double aatom = emass / pmass;

  // set energy loss per turn manually
  // TODO: implement radiation update of twiss
  twheader["U0"] = 174e3;
  // update twiss header with long parameters
  ste_longitudinal::updateTwissHeaderLong(twheader, h, v, aatom, sigs);

  int nMacro = 4096;
  double betx = twheader["LENGTH"] / (twheader["Q1"] * 2.0 * pi);
  double bety = twheader["LENGTH"] / (twheader["Q2"] * 2.0 * pi);
  double coupling = 0.05;
  double ex = 5e-9;
  double ey = coupling * ex;

  std::vector<std::vector<double>> distribution =
      ste_random::GenerateDistributionMatched(nMacro, betx, ex, bety, ey, h, v,
                                              twheader, seed);
  /*
std::vector<double> emit =
ste_global::CalculateEmittance(distribution, betx, bety);

std::printf("%-20s %16.8e\n", "ex", ex);
std::printf("%-20s %16.8e\n", "ey", ey);
std::printf("%-20s %16.8e\n", "betx", betx);
std::printf("%-20s %16.8e\n", "bety", bety);
ste_output::printVector(emit);
*/
  // test calc emittance
  std::printf("%-30s", "test_calc_emit ");
  if (test_calc_emit(distribution, betx, bety)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();
  return 0;
}