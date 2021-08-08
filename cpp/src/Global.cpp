#include <algorithm>
#include <ibs>
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

void betatronUpdate(std::vector<std::vector<double>> &distribution,
                    std::map<std::string, double> tw, double coupling,
                    double K2L, double K2SL) {

  double qx = tw["Q1"];
  double qy = tw["Q2"];
  double ksix = tw["DQ1"];
  double ksiy = tw["DQ2"];

  double psix = 2.0 * pi * qx;
  double psiy = 2.0 * pi * qy;
  std::for_each(distribution.begin(), distribution.end(),
                [&tw, &psix, &ksix, &psiy, &ksiy, &coupling, &K2L,
                 &K2SL](std::vector<double> &particle) {
                  // rotation in x-px plane
                  float psi1 = psix + particle[5] * ksix;
                  float a11 = cos(psi1);
                  float a12 = sin(psi1);

                  particle[0] = particle[0] * a11 + particle[1] * a12;
                  particle[1] = particle[1] * a11 - particle[0] * a12;

                  // rotation in y-py plane
                  double psi2 = psiy + particle[5] * ksiy;
                  a11 = cos(psi2);
                  a12 = sin(psi2);

                  particle[2] = particle[2] * a11 + particle[3] * a12;
                  particle[3] = particle[3] * a11 - particle[2] * a12;

                  // now have dqmin part - coupling between x and y
                  particle[1] += coupling * particle[2];
                  particle[3] += coupling * particle[0];

                  // thin sextupole kick
                  particle[1] +=
                      0.5 * K2L * (pow(particle[0], 2) - pow(particle[2], 2)) -
                      K2SL * (particle[0] * particle[1]);
                  particle[3] +=
                      0.5 * K2SL * (pow(particle[0], 2) - pow(particle[2], 2)) +
                      K2L * (particle[0] * particle[1]);
                });
}
} // namespace ste_global