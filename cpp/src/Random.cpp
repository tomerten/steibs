#include "../include/ste_bits/Random.hpp"
#include "../include/ste_bits/Output.hpp"
// #include <CL/cl.h>
#include "../include/ste_bits/Longitudinal.hpp"
#include <ibs>
#include <iostream>
#include <math.h>
#include <vector>

namespace ste_random {

#define MBIG 1000000000
#define MSEED 161803398
#define MZ 0
#define FAC (1.0 / MBIG)

/*
================================================================================
================================================================================
RANDOM UNIFORM GENERATOR [0-1]
================================================================================
REFERENCE:
    - Numerical recipes in C
================================================================================
Arguments:
----------
    - *int : seed of the generator

Returns:
--------
    long double
        random value
================================================================================
================================================================================
*/
double ran3(int *idum) {
  static int inext, inextp;
  static int ma[56];
  static int iff = 0;
  int mj, mk;
  int i, ii, k;
  if (*idum < 0 || iff == 0) {
    iff = 1;
    mj = MSEED - (*idum < 0 ? -*idum : *idum);
    mj %= MBIG;
    ma[55] = mj;
    mk = 1;
    for (i = 1; i <= 54; i++) {
      ii = (21 * i) % 55;
      ma[ii] = mk;
      mk = mj - mk;
      if (mk < MZ)
        mk += MBIG;
      mj = ma[ii];
    }
    for (k = 1; k <= 4; k++)
      for (i = 1; i <= 55; i++) {
        ma[i] -= ma[1 + (i + 30) % 55];
        if (ma[i] < MZ)
          ma[i] += MBIG;
      }
    inext = 0;
    inextp = 31;
    *idum = 1;
  }
  if (++inext == 56)
    inext = 1;
  if (++inextp == 56)
    inextp = 1;
  mj = ma[inext] - ma[inextp];
  if (mj < MZ)
    mj += MBIG;
  ma[inext] = mj;
  return (mj * FAC);
}

#undef MBIG
#undef MSEED
#undef MZ
#undef FAC

std::vector<double> BiGaussian4D(const double betax, const double ex,
                                 const double betay, const double ey,
                                 int seed) {
  std::vector<double> out;

  static double ampx, ampy, amp, r1, r2, facc;
  static double x, px, y, py;

  // 1 sigma rms beam sizes using average ring betas
  ampx = sqrt(betax * ex);
  ampy = sqrt(betay * ey);

  // generate bi-gaussian distribution in the x-px phase-space
  do {
    r1 = 2 * ran3(&seed) - 1;
    r2 = 2 * ran3(&seed) - 1;
    amp = r1 * r1 + r2 * r2;
  } while ((amp >= 1) || (amp <= 3.e-6));

  facc =
      sqrt(-2 * log(amp) /
           amp); // transforming [-1,1] uniform to gaussian - inverse transform

  x = ampx * r1 * facc;  // scaling the gaussian
  px = ampx * r2 * facc; // scaling the gaussian

  // generate bi-gaussian distribution in the y-py phase-space
  do {
    r1 = 2 * ran3(&seed) - 1;
    r2 = 2 * ran3(&seed) - 1;
    amp = r1 * r1 + r2 * r2;
  } while ((amp >= 1) || (amp <= 3.e-6));

  facc =
      sqrt(-2 * log(amp) /
           amp); // transforming [-1,1] uniform to gaussian - inverse transform

  y = ampy * r1 * facc;  // scaling the gaussian
  py = ampy * r2 * facc; // scaling the gaussian

  out.push_back(x);
  out.push_back(px);
  out.push_back(y);
  out.push_back(py);

  return out;
}

std::vector<double> BiGaussian6D(double betax, double ex, double betay,
                                 double ey,
                                 std::vector<double> &harmonicNumbers,
                                 std::vector<double> &rfVoltages,
                                 std::map<std::string, double> &twissheadermapL,
                                 int seed) {
  double h0 = harmonicNumbers[0];
  double tauhat = twissheadermapL["tauhat"];
  double omega = twissheadermapL["omega"];
  double ampt = twissheadermapL["sigs"] / clight;
  double ts = twissheadermapL["phis"] / 180.0 * pi / (h0 * omega);
  int npi = int(twissheadermapL["phis"]) / 360;
  double tperiod = 2 * pi / (h0 * omega);
  double ts2 = (180.0 - twissheadermapL["phis"]) / 180.0 * pi / (h0 * omega) +
               double(npi) * tperiod;

  /*
  std::printf("%-20s %16.8e\n", "h0", h0);
  std::printf("%-20s %16.8e\n", "omega", omega);
  std::printf("%-20s %16.8e\n", "pi", pi);
  std::printf("%-20s %16.8e\n", "phis", twissheadermapL["phis"]);
  */
  std::vector<double> out;
  out = BiGaussian4D(betax, ex, betay, ey, seed);

  // adding two zeros
  out.push_back(0.0);
  out.push_back(0.0);

  // ste_output::printVector(out);
  // longitudinal matching
  double r1, r2, amp, facc, tc, pc, ham;
  tc = (omega * twissheadermapL["eta"] * h0);
  pc = (omega * twissheadermapL["CHARGE"]) /
       (2.0 * pi * twissheadermapL["PC"] * 1.0e9 * twissheadermapL["betar"]);

  double delta = twissheadermapL["sige"]; // 0.0; // sync particle
  // std::printf("%-20s %16.8e\n", "delta", delta);

  double hammax = ste_longitudinal::Hamiltonian(
      twissheadermapL, harmonicNumbers, rfVoltages, tc, ts2, 0.0);
  // std::printf("%-20s %16.8e\n", "hammax", hammax);
  int looper = 0;
  do {
    looper++;
    // std::printf("%i\n", looper);
    r1 = 2 * ran3(&seed) - 1;
    r2 = 2 * ran3(&seed) - 1;
    amp = r1 * r1 + r2 * r2;
    if (amp >= 1)
      continue;

    facc = sqrt(-2 * log(amp) / amp);
    // std::printf("%-20s %16.8e\n", "out4", out[4]);
    out[4] = ts + ampt * r1 * facc;
    /*
    std::printf("%-20s %16.8e\n", "ts", ts);
    std::printf("%-20s %16.8e\n", "ampt", ampt);
    std::printf("%-20s %16.8e\n", "r1", r1);
    std::printf("%-20s %16.8e\n", "facc", facc);
    std::printf("%-20s %16.8e\n", "out4", out[4]);
    */
    if (abs(out[4] - ts) >= abs(ts - ts2))
      continue;

    if (looper >= 10) {
      std::printf("%-20s %16.8e\n", "out4-ts", abs(out[4] - ts));
      std::printf("%-20s %16.8e\n", "tauhat", abs(ts - ts2));
    }

    out[5] = twissheadermapL["sige"] * r2 * facc;
    ham = ste_longitudinal::Hamiltonian(twissheadermapL, harmonicNumbers,
                                        rfVoltages, tc, out[4], out[5]);
    /*std::printf("%-20s %16.8e\n", "hammax", hammax);
    std::printf("%-20s %16.8e\n", "ham", ham);
    */
  } while ((amp >= 1) || (ham > hammax) || (abs(out[4] - ts) >= abs(ts - ts2)));

  /// ste_output::printVector(out);
  return out;
}

std::vector<double> BiGaussian6DLongMatched(
    double betax, double ex, double betay, double ey,
    std::vector<double> &harmonicNumbers, std::vector<double> &rfVoltages,
    std::map<std::string, double> &twissheadermapL, int seed) {

  double h0 = harmonicNumbers[0];
  double tauhat = twissheadermapL["tauhat"];
  double omega = twissheadermapL["omega"];
  double ampt = twissheadermapL["sigs"] / clight;
  // Max value Hamiltonian that is stable
  // is, with the sign convention used, left of the ham contour
  // at 180-phis (The Ham rises lin to the right.)
  double ts = twissheadermapL["phis"] / 180.0 * pi / (h0 * omega);
  int npi = int(twissheadermapL["phis"]) / 360;
  double tperiod = 2 * pi / (h0 * omega);
  double ts2 = (180.0 - twissheadermapL["phis"]) / 180.0 * pi / (h0 * omega) +
               double(npi) * tperiod;
  double delta = twissheadermapL["sige"];

  std::vector<double> out;
  out = BiGaussian4D(betax, ex, betay, ey, seed);

  // adding two zeros
  out.push_back(0.0);
  out.push_back(0.0);

  double r1, r2, amp, facc, tc, pc, ham, hammin;
  tc = (omega * twissheadermapL["eta"] * h0);
  pc = (omega * twissheadermapL["CHARGE"]) /
       (2.0 * pi * twissheadermapL["PC"] * 1.0e9 * twissheadermapL["betar"]);

  // max Hamiltonian
  double hammax = ste_longitudinal::Hamiltonian(
      twissheadermapL, harmonicNumbers, rfVoltages, tc, ts2, 0.0);
  // std::printf("%-20s %16.8e\n", "hammax", ts);
  // std::printf("%-20s %16.8e\n", "ts", ts);
  // std::printf("%-20s %16.8e\n", "ts2", ts2);
  // select valid t values
  do {
    // looper++;
    // std::printf("%i\n", looper);
    r1 = 2 * ran3(&seed) - 1;
    r2 = 2 * ran3(&seed) - 1;
    amp = r1 * r1 + r2 * r2;
    if (amp >= 1)
      continue;

    facc = sqrt(-2 * log(amp) / amp);
    // std::printf("%-20s %16.8e\n", "out4", out[4]);
    out[4] = ts + ampt * r1 * facc;
    /*
    std::printf("%-20s %16.8e\n", "ts", ts);
    std::printf("%-20s %16.8e\n", "ampt", ampt);
    std::printf("%-20s %16.8e\n", "r1", r1);
    std::printf("%-20s %16.8e\n", "facc", facc);
    std::printf("%-20s %16.8e\n", "out4", out[4]);
    std::printf("%-20s %16.8e\n", "out4-ts", out[4] - ts);
    std::printf("%-20s %16.8e\n", "tauhat", tauhat);
    */
    if (abs(out[4] - ts) >= abs(ts - ts2))
      continue;

    // min Hamiltonian
    hammin = ste_longitudinal::Hamiltonian(twissheadermapL, harmonicNumbers,
                                           rfVoltages, tc, out[4], 0.0);
    // std::printf("%-20s %16.8e\n", "hammin", hammin);
    // std::printf("%-20s %16.8e\n", "hammax", hammax);

  } while ((hammin > hammax) || (abs(out[4] - ts) >= abs(ts - ts2)));

  // select matched deltas
  do {
    // looper++;
    // std::printf("%i\n", looper);
    r1 = 2 * ran3(&seed) - 1;
    r2 = 2 * ran3(&seed) - 1;
    amp = r1 * r1 + r2 * r2;

    if (amp >= 1)
      continue;

    facc = sqrt(-2 * log(amp) / amp);
    out[5] = twissheadermapL["sige"] * r2 * facc;
    ham = ste_longitudinal::Hamiltonian(twissheadermapL, harmonicNumbers,
                                        rfVoltages, tc, out[4], out[5]);
    // std::printf("%-20s %16.8e\n", "hammin", hammin);
    // std::printf("%-20s %16.8e\n", "ham", ham);
    // std::printf("%-20s %16.8e\n", "hammax", hammax);
  } while ((ham < hammin) || (ham > hammax));

  return out;
}

std::vector<std::vector<double>>
GenerateDistribution(int nMacro, double betax, double ex, double betay,
                     double ey, std::vector<double> &harmonicNumbers,
                     std::vector<double> &rfVoltages,
                     std::map<std::string, double> &twissheadermapL, int seed) {
  std::vector<std::vector<double>> out;

  for (int i = 0; i < nMacro; i++) {
    out.push_back(BiGaussian6D(betax, ex, betay, ey, harmonicNumbers,
                               rfVoltages, twissheadermapL, seed));
  }
  return out;
}

std::vector<std::vector<double>> GenerateDistributionMatched(
    int nMacro, double betax, double ex, double betay, double ey,
    std::vector<double> &harmonicNumbers, std::vector<double> &rfVoltages,
    std::map<std::string, double> &twissheadermapL, int seed) {
  std::vector<std::vector<double>> out;

  for (int i = 0; i < nMacro; i++) {
    out.push_back(BiGaussian6DLongMatched(betax, ex, betay, ey, harmonicNumbers,
                                          rfVoltages, twissheadermapL, seed));
  }
  return out;
}
} // namespace ste_random