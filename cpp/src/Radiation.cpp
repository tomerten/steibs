#include "../include/ste_bits/Random.hpp"
#include <algorithm>
#include <complex>
#include <ibs>
#include <map>
#include <math.h>
#include <string>
#include <vector>
namespace ste_radiation {

std::map<std::string, double>
radiationIntegrals(std::map<std::string, std::vector<double>> &twiss) {
  std::map<std::string, double> integrals;

  double *radint = RadiationDampingLattice(twiss);
  // printradint(radint);

  integrals["I2"] = radint[1];
  integrals["I3"] = radint[2];
  integrals["I4x"] = radint[3];
  integrals["I4y"] = radint[4];
  integrals["I5x"] = radint[5];
  integrals["I5y"] = radint[6];

  return integrals;
}

std::map<std::string, double>
radiationEquilib(std::map<std::string, double> &twheader) {
  const double electron_volt_joule_relationship = 1.602176634e-19;
  const double hbar = 1.0545718176461565e-34;
  std::map<std::string, double> output;

  // base quantities
  double gamma = twheader["GAMMA"];
  double gammatr = twheader["GAMMATR"];
  double p0 = twheader["PC"] * 1.0e9;
  double len = twheader["LENGTH"] * 1.0;
  double restE = twheader["MASS"] * 1.0e9;
  double charge = twheader["CHARGE"] * 1.0;
  double aatom = twheader["aatom"];
  double qs = twheader["qs"];
  double omega = twheader["omega"];
  double q1 = twheader["Q1"];

  // use madx rad integrals
  double I1 = twheader["SYNCH_1"];
  double I2 = twheader["SYNCH_2"];
  double I3 = twheader["SYNCH_3"];
  double I4x = twheader["SYNCH_4"];
  double I5x = twheader["SYNCH_5"];

  double I4y = 0.0;
  double I5y = 0.0;

  // derived quantities
  double pradius = ParticleRadius(charge, aatom);
  double CalphaEC =
      pradius * clight / (3.0 * restE * restE * restE) * (p0 * p0 * p0 / len);

  // transverse partition numbers
  double jx = 1.0 - I4x / I2;
  double jy = 1.0 - I4y / I2;
  double alphax = 2.0 * CalphaEC * I2 * jx;
  double alphay = 2.0 * CalphaEC * I2 * jy;
  double alphas = 2.0 * CalphaEC * I2 * (jx + jy);

  // mc**2 expressed in Joule to match units of cq
  double mass = restE * electron_volt_joule_relationship;
  double cq = 55.0 / (32.0 * sqrt(3.0)) * (hbar * clight) / mass;

  double sigE0E2 = cq * gamma * gamma * I3 / (2.0 * I2 + I4x + I4y);
  // ! = deltaE/E_0 see wiedemann p. 302,
  // and Wolski: E/(p0*c) - 1/beta0 = (E - E0)/(p0*c) = \Delta E/E0*beta0 with
  // E0 = p0*c/beta0 therefore:
  double betar = BetaRelativisticFromGamma(gamma);
  double dpop = dee_to_dpp(sqrt(sigE0E2), betar);
  double sigs = dpop * len * eta(gamma, gammatr) / (2 * pi * qs);
  double exinf = cq * gamma * gamma * I5x / (jx * I2);
  double eyinf = cq * gamma * gamma * I5y / (jy * I2);

  double betaAvg = len / (q1 * 2.0 * pi);

  eyinf = (eyinf == 0.0) ? cq * betaAvg * I3 / (2.0 * jy * I2) : eyinf;

  output["taux"] = 1.0 / alphax;
  output["tauy"] = 1.0 / alphay;
  output["taus"] = 1.0 / alphas;
  output["exinf"] = exinf;
  output["eyinf"] = eyinf;
  output["sigeoe2"] = sigE0E2;
  output["sigsinf"] = sigs;
  output["jx"] = jx;
  output["jy"] = jy;

  return output;
}

double RadiationLossesPerTurn(std::map<std::string, double> &twiss) {
  double gamma = twiss["GAMMA"];
  double p0 = twiss["PC"];
  double len = twiss["LENGTH"];
  double mass = twiss["MASS"];
  double charge = twiss["CHARGE"];
  double aatom = twiss["aatom"];
  double I2 = twiss["SYNCH_2"];

  double particle_radius = ParticleRadius(charge, aatom);
  double cgamma = (4.0 * pi / 3.0) * (particle_radius / (mass * mass * mass));
  double trev = twiss["trev"];

  return (clight * cgamma) / (2.0 * pi * len) * p0 * p0 * p0 * p0 * I2 * 1.0e9 *
         trev;
}

void RadUpdate(std::vector<std::vector<double>> &distribution,
               std::map<std::string, double> &tw,
               std::map<std::string, double> &radparam, int &seed) {
  std::for_each(
      distribution.begin(), distribution.end(),
      [&tw, &radparam, &seed](std::vector<double> &particle) {
        /*
      for (map<string, double>::const_iterator it = radparam.begin();
           it != radparam.end(); ++it) {
        std::printf("%-20s %16.8e\n", it->first.c_str(), it->second);
      }*/
        double trev = tw["trev"];
        double timeratio = tw["timeratio"];
        std::printf("test %16.6e\n", radparam["taus"]);
        // timeratio is real machine turns over per simulation turn
        double coeffdecaylong = 1.0 - ((trev / radparam["taus"]) * timeratio);
        double coeffexcitelong = radparam["sigsinf"] / clight * sqrt(3.0) *
                                 sqrt(1.0 - coeffdecaylong * coeffdecaylong);
        std::printf("%-30s %24.16e\n", "taus", radparam["taus"]);
        std::printf("%-30s %24.16e\n", "coefdecaylong ", coeffdecaylong);
        std::printf("%-30s %24.16e\n", "coefexcitelong ", coeffexcitelong);

        // the damping time is for EMITTANCE, therefore need to multiply by 2
        double coeffdecayx =
            1.0 - ((trev / (2.0 * radparam["taux"])) * timeratio);
        double coeffdecayy =
            1.0 - ((trev / (2.0 * radparam["tauy"])) * timeratio);
        std::printf("%-30s %24.16e\n", "coeffdecayx ", coeffdecayx);

        // exact     coeffgrow= sigperp*sqrt(3.)*sqrt(1-coeffdecay**2)
        // but trev << tradperp so
        double coeffgrowx =
            radparam["exinf"] * sqrt(3.) * sqrt(1.0 - pow(coeffdecayx, 2));
        double coeffgrowy =
            radparam["eyinf"] * sqrt(3.) * sqrt(1.0 - pow(coeffdecayy, 2));

        if (!((radparam["taux"] < 0.0) || (radparam["tauy"] < 0.0))) {
          std::printf("init %24.16e\n", particle[0]);
          particle[0] = coeffdecayx * particle[0] +
                        coeffgrowx * (2.0 * ste_random::ran3(&seed) - 1.0);
          std::printf("init %24.16e\n", particle[0]);
          particle[1] = coeffdecayx * particle[1] +
                        coeffgrowx * (2.0 * ste_random::ran3(&seed) - 1.0);
          particle[2] = coeffdecayy * particle[2] +
                        coeffgrowy * (2.0 * ste_random::ran3(&seed) - 1.0);
          particle[3] = coeffdecayy * particle[3] +
                        coeffgrowy * (2.0 * ste_random::ran3(&seed) - 1.0);
          particle[5] = coeffdecaylong * particle[5] +
                        coeffexcitelong * (2.0 * ste_random::ran3(&seed) - 1.0);
        };
      });
}
} // namespace ste_radiation