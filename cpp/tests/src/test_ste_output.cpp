#include <ste>

int main() {
  std::vector<double> PrintTestVector;
  double a = 1.0;
  for (int i = 0; i < 6; i++) {
    a += 1.0;
    PrintTestVector.push_back(a);
  }
  ste_output::printVector(PrintTestVector);
  return 0;
}