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

bool test_single_bigaussian4D(double betax, double ex, double betay, double ey,
                              int seed) {
  std::vector<double> testVector;
  testVector = ste_random::BiGaussian4D(betax, ex, betay, ey, seed);
  ste_output::printVector(testVector);
  return false;
}

int main() {
  // set seed
  int seed = 123456;
  std::string test[] = {"test_single_ran3"};

  // test single random number
  std::printf("test_single_ran3 ");
  if (test_single_ran3(seed)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test single bigaussian4d
  test_single_bigaussian4D(1.0, 1.0e-9, 2.0, 1.0e-10, seed);
  test_single_bigaussian4D(1.0, 1.0e-9, 2.0, 1.0e-10, seed);
  return 0;
}