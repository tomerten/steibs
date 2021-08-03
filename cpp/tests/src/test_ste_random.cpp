#include <ibs>
#include <iomanip>
#include <iostream>
#include <ste>

bool test_single_ran3(int seed) {
  long double rndVal = ste_random::ran3(&seed);
  if (rndVal - 0.386572938 < 1.0e-9) {
    return true;
  }
  return false;
}

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

bool test_single_bigaussian4D(double betax, double ex, double betay, double ey,
                              int seed) {
  std::vector<double> expected, testVector;
  expected = {4.42441e-05, -3.04531e-05, -1.04727e-05, 1.16399e-05};
  testVector = ste_random::BiGaussian4D(betax, ex, betay, ey, seed);

  return RequireVectorEquals(testVector, expected);
}

bool test_single_bigaussian6D(double betax, double ex, double betay, double ey,
                              std::map<std::string, double> &twissheader,
                              int seed) {
  std::vector<double> expected, testVector, h, v;
  h.push_back(400.0);
  v.push_back(-1.5e6);

  expected = {8.26179e-08,  6.54118e-05, 1.61556e-05,
              -5.22136e-06, 9.86857e-10, -0.000713646};
  testVector =
      ste_random::BiGaussian6D(betax, ex, betay, ey, h, v, twissheader, seed);

  return RequireVectorEquals(testVector, expected);
}

int main() {
  // set seed
  int seed = 123456;

  string twissfilename = "../src/b2_design_lattice_1996.twiss";
  map<string, double> twissheadermap;
  twissheadermap = GetTwissHeader(twissfilename);
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
  twissheadermap["U0"] = 174e3;
  // update twiss header with long parameters
  ste_longitudinal::updateTwissHeaderLong(twissheadermap, h, v, aatom, sigs);

  // test single random number
  std::printf("%-30s", "test_single_ran3 ");
  if (test_single_ran3(seed)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test bigaussian4d
  std::printf("%-30s", "test_single_bigaussian4d ");
  if (test_single_bigaussian4D(1.0, 1.0e-9, 2.0, 1.0e-10, seed)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test bigaussian6d
  std::printf("%-30s", "test_single_bigaussian6d ");
  if (test_single_bigaussian6D(1.0, 1.0e-9, 2.0, 1.0e-10, twissheadermap,
                               seed)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();
  // test single bigaussian4d
  // test_single_bigaussian4D(1.0, 1.0e-9, 2.0, 1.0e-10, seed);
  // test_single_bigaussian4D(1.0, 1.0e-9, 2.0, 1.0e-10, seed);
  return 0;
}