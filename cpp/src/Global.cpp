#include <algorithm>
#include <map>
#include <math.h>
#include <string>
#include <vector>

namespace ste_global {

std::vector<double> CalculateEmittance(std::vector<std::vector<double>> arr,
                                       double betx, double bety) {
  std::vector<double> out, avg;

  for (int i = 0; i < arr[0].size(); i++) {
    out.push_back(0.0);
    avg.push_back(0.0);
  }

  // calculate averages
  std::for_each(arr.begin(), arr.end(), [&](std::vector<double> v) {
    for (int i = 0; i < arr[0].size(); i++) {
      avg[i] += v[i] / arr.size();
    }
  });

  // subtract avg, square and sum
  std::for_each(arr.begin(), arr.end(), [&](std::vector<double> v) {
    for (int i = 0; i < arr[0].size(); i++) {
      out[i] += ((v[i] - avg[i]) * (v[i] - avg[i])) / arr.size();
    }
  });

  out[0] /= betx;
  out[2] /= bety;
  out[4] = sqrt(out[4]);
  out[5] = sqrt(out[5]);

  return out;
}
} // namespace ste_global