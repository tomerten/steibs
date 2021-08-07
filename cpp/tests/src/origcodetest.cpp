#include <boost/math/tools/roots.hpp>
#include <ibs>
#include <ste>
//#include <boost/math/tools/roots.hpp>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <math.h>
#include <string>

template <class T> struct synchronousPhaseFunctor {
  synchronousPhaseFunctor(T const &target, std::vector<double> &voltages,
                          std::vector<double> &harmonicNumbers, float charge)
      : U0(target), volts(voltages), hs(harmonicNumbers), ch(charge) {}
  std::tuple<double, double> operator()(T const &phi) {
    T volt1, volt2, volt3;
    T dvolt1, dvolt2, dvolt3;
    volt1 = ch * volts[0] * sin(phi);
    volt2 = ch * volts[1] * sin((hs[1] / hs[0]) * phi);
    volt3 = ch * volts[2] * sin((hs[2] / hs[0]) * phi);

    dvolt1 = ch * volts[0] * cos(phi);
    dvolt2 = ch * volts[1] * (hs[1] / hs[0]) * cos((hs[1] / hs[0]) * phi);
    dvolt3 = ch * volts[2] * (hs[2] / hs[0]) * cos((hs[2] / hs[0]) * phi);
    std::tuple<double, double> out = {volt1 + volt2 + volt3 - U0,
                                      dvolt1 + dvolt2 + dvolt3};
    return out;
  }

private:
  T U0;
  std::vector<double> volts;
  std::vector<double> hs;
  double ch;
};

template <class T>
T synchronousPhaseFunctorDeriv(T x, std::vector<double> &voltages,
                               std::vector<double> &harmnumbers, double charge,
                               T guess, T min, T max) {
  // return cube root of x using 1st derivative and Newton_Raphson.
  using namespace boost::math::tools;

  const int digits =
      std::numeric_limits<T>::digits; // Maximum possible binary digits accuracy
                                      // for type T.
  int get_digits = static_cast<int>(
      digits * 0.6); // Accuracy doubles with each step, so stop when we have
                     // just over half the digits correct.
  const boost::uintmax_t maxit = 20;
  boost::uintmax_t it = maxit;
  T result = newton_raphson_iterate(
      synchronousPhaseFunctor<T>(x, voltages, harmnumbers, charge), guess, min,
      max, get_digits, it);
  return result;
}
double pcoeff(double voltage, double angularFreq, double p0,
              double betaRelativistic, double charge) {
  return (angularFreq * voltage * charge) / (2.0 * pi * p0 * betaRelativistic);
}

double HamiltonianTripleRf(double tcoeff, std::vector<double> &voltages,
                           std::vector<double> &harmonicNumbers,
                           double phiSynchronous, double t, double delta,
                           double omega0, double p0, double betaRelativistic,
                           double charge) {
  double kinetic, potential1, potential2, potential3;
  kinetic = 0.5 * tcoeff * pow(delta, 2);

  std::printf("%-30s %16.8e\n", "kinetic", kinetic);
  potential1 = pcoeff(voltages[0], omega0, p0, betaRelativistic, charge) *
               (cos(harmonicNumbers[0] * omega0 * t) - cos(phiSynchronous) +
                (harmonicNumbers[0] * omega0 * t - phiSynchronous) *
                    sin(phiSynchronous));
  std::printf("%-30s %16.8e\n", "pcoeff0",
              pcoeff(voltages[0], omega0, p0, betaRelativistic, charge));
  std::printf("%-30s %16.8e\n", "phase0", harmonicNumbers[0] * omega0 * t);
  std::printf("%-30s %16.8e\n", "potential1", potential1);

  potential2 =
      pcoeff(voltages[1], omega0, p0, betaRelativistic, charge) *
      (harmonicNumbers[0] / harmonicNumbers[1]) *
      (cos(harmonicNumbers[1] * omega0 * t) -
       cos(harmonicNumbers[1] * phiSynchronous / harmonicNumbers[0]) +
       (harmonicNumbers[1] * omega0 * t -
        harmonicNumbers[1] * phiSynchronous / harmonicNumbers[0]) *
           sin(harmonicNumbers[1] * phiSynchronous / harmonicNumbers[0]));

  potential3 =
      pcoeff(voltages[2], omega0, p0, betaRelativistic, charge) *
      (harmonicNumbers[0] / harmonicNumbers[2]) *
      (cos(harmonicNumbers[2] * omega0 * t) -
       cos(harmonicNumbers[2] * phiSynchronous / harmonicNumbers[0]) +
       (harmonicNumbers[2] * omega0 * t -
        harmonicNumbers[2] * phiSynchronous / harmonicNumbers[0]) *
           sin(harmonicNumbers[2] * phiSynchronous / harmonicNumbers[0]));

  return kinetic + potential1 + potential2 + potential3;
};

double synchrotronTune(std::map<string, double> &twissheadermap,
                       std::vector<double> h, std::vector<double> v) {
  double p0 = twissheadermap["PC"] * 1.0e9;
  double phis = twissheadermap["phis"];
  double charge = twissheadermap["CHARGE"];
  double Omega2 =
      (h[0] * twissheadermap["eta"] * charge) / (2.0 * pi * p0) *
      (v[0] * cos(phis) + v[1] * (h[1] / h[0]) * cos((h[1] / h[0]) * phis) +
       v[2] * (h[2] / h[0]) * cos((h[2] / h[0]) * phis));
  return sqrt(abs(Omega2));
};

int main() {
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

  double energyLostPerTurn = 174000;    // radation losses per turn per particle
  double acceleratorLength = 240.00839; // length in meter
  double gammar = twissheadermap["GAMMA"]; // relativistic gamma
  // double eta = 0.0007038773471 -
  //          1 / pow(gammar, 2); // slip factor approx alpha - 1/ gammar**2
  double betar = twissheadermap["betar"]; // relativistic beta
  double trev = twissheadermap["trev"];
  double h0 = 400.0;
  double h1 = 400; // 1200.0;
  double h2 = 400; // 1400.0;
  double v0 = -1.5e6;
  double v1 = 0.0; // 20.0e6;
  double v2 = 0.0; // 17.14e6;
  double omega0 = (2 * pi) / trev;
  double p0 = 1.7e9;
  double charge = -1.0;
  std::vector<double> hnumbers = {h0, h1, h2};
  std::vector<double> voltages = {v0, v1, v2};

  double search1 =
      trev * h0 * omega0 /
      (8 * max(max(h0, h1), h2)); // give positive offset to find upstream
                                  // root and not downstream root
  double search2 = trev * h0 * omega0 / (8 * max(max(h0, h1), h2)) -
                   trev * h0 * omega0 / min(min(h0, h1), h2);
  double searchWidth = trev * h0 * omega0 / (2 * max(max(h0, h1), h2));
  ste_output::cyan();
  std::printf("%-30s : %16.6f\n", "Search1", search1 / pi * 180);
  std::printf("%-30s : %16.6f\n", "Search2", search2 / pi * 180);
  std::printf("%-30s : %16.6f\n", "Width", searchWidth / pi * 180);
  ste_output::reset();

  double synchronousPhase0 = synchronousPhaseFunctorDeriv(
      energyLostPerTurn, voltages, hnumbers, charge, search1,
      search1 - searchWidth, search1 + searchWidth);
  double synchronousPhase1 = synchronousPhaseFunctorDeriv(
      energyLostPerTurn, voltages, hnumbers, charge, search2,
      search2 - searchWidth, search2 + searchWidth);

  ste_output::blue();
  std::printf("%-30s : %16.6f %16.6f %16.6e \n", "synchronous phase", search1,
              synchronousPhase0 / pi * 180, synchronousPhase0 / (h0 * omega0));
  std::printf("%-30s : %16.6f %16.6f %16.6e \n", "synchronous phase", search2,
              synchronousPhase1 / pi * 180, synchronousPhase1 / (h0 * omega0));
  ste_output::yellow();
  std::printf("%-30s : %16.6f %16.6f %16.6e\n", "synchronous phase", 173.0,
              twissheadermap["phis"],
              twissheadermap["phis"] / 180.0 * pi / (h0 * omega0));
  ste_output::reset();

  std::cout << "Find next extremum of Hamiltonian" << std::endl;

  double synchronousPhase0Next = synchronousPhaseFunctorDeriv(
      energyLostPerTurn, voltages, hnumbers, charge, search1 + searchWidth,
      search1 + searchWidth / 2, search1 + 2 * searchWidth);
  double synchronousPhase1Next = synchronousPhaseFunctorDeriv(
      energyLostPerTurn, voltages, hnumbers, charge, search2 + searchWidth,
      search2 + searchWidth / 2, search2 + 2 * searchWidth);

  ste_output::blue();
  std::printf("%-30s : %16.6f %16.6f %16.6e \n", "synchronous phase next",
              search1 + searchWidth, synchronousPhase0Next / pi * 180,
              synchronousPhase0Next / (h0 * omega0));
  std::printf("%-30s : %16.6f %16.6f %16.6e \n", "synchronous phase next ",
              search2 + searchWidth, synchronousPhase1Next / pi * 180,
              synchronousPhase1Next / (h0 * omega0));
  ste_output::yellow();
  std::printf("%-30s : %16.6f\n", "synchronous phase", twissheadermap["phis"]);
  ste_output::reset();

  std::printf("%s", "\nHamiltonians\n");
  ste_output::blue();
  double tc = ste_longitudinal::tcoeff(twissheadermap, h0);
  double tcval = tc / (h0 * omega0);
  double ohammax = HamiltonianTripleRf(
      tcval, voltages, hnumbers, synchronousPhase1,
      synchronousPhase1Next / (h0 * omega0), 0.0, omega0, p0, betar, charge);
  double hammax =
      ste_longitudinal::Hamiltonian(twissheadermap, hnumbers, voltages, tc,
                                    synchronousPhase1Next / (h0 * omega0), 0.0);
  std::printf("%-30s : %16.6f \n", "tcoeff", tcval);
  std::printf("%-30s : %16.6f \n", "ohammax", ohammax);
  ste_output::yellow();
  std::printf("%-30s : %16.6f \n", "hammax", hammax);
  ste_output::reset();

  std::cout << "Synchrotron tune : "
            << synchrotronTune(twissheadermap, hnumbers, voltages) << std::endl;
  std::cout << "Synchrotron tune (Hz): "
            << synchrotronTune(twissheadermap, hnumbers, voltages) * omega0 /
                   (2.0 * pi)
            << std::endl;
  std::printf("%-30s : %16.6f \n", "syncTune",
              synchrotronTune(twissheadermap, hnumbers, voltages));
  std::printf("%-30s : %16.6f \n", "syncTune Hz",
              synchrotronTune(twissheadermap, hnumbers, voltages) * omega0 /
                  (2.0 * pi));
  ste_output::yellow();
  std::printf("%-30s : %16.6f \n", "syncTune", twissheadermap["qs"]);
  std::printf("%-30s : %16.6f \n", "syncTune Hz",
              twissheadermap["qs"] * omega0 / (2.0 * pi));
  ste_output::reset();

  std::printf("%16.6e  %16.6e %16.6e %16.6e %16.6e\n", synchronousPhase0Next,
              synchronousPhase0,
              abs((synchronousPhase0Next - synchronousPhase0) / pi * 180.0), h0,
              omega0);
  std::printf("%-30s : %16.6e \n", "tauhat",
              abs((synchronousPhase0Next - synchronousPhase0) / (h0 * omega0)));
  ste_output::yellow();
  std::printf("%-30s : %16.6e \n", "tauhat", twissheadermap["tauhat"]);
  std::printf("%-30s : %16.6e \n", "tauhat",
              (twissheadermap["phis"] - (twissheadermap["phis"] - 180.0)) /
                  180.0 * pi / (h0 * omega0));
  ste_output::reset();
  return 0;
}