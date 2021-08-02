#include <ibs>
#include <map>
#include <string>
#include <vector>

namespace ste_longitudinal {

/*
================================================================================
================================================================================
METHOD TO CALCULATE NECESSARY LONGITUDINAL PARAMETERS FOR REUSE DURING
SIMULATION.
================================================================================
AUTHORS:
    - TOM MERTENS

HISTORY:
    - INITIAL VERSION 02/08/2021

================================================================================
Arguments:
----------
    - std::map<std::string, double> twissheader
        dictionary containing Twiss header data (ref: IBSLib -> GetTwissHeader)
    - std::vector<double> harmonicNumbers
        list of harmonic numbers of the RF systems
    - std::vector<double> rfVoltages
        list of the corresponding RF voltages
    - double aatom
        Atomic mass number (for electrons use emass / pmass)
    - double sigs
        Bunch length

================================================================================
================================================================================
*/
void updateTwissHeaderLong(std::map<std::string, double> &twissheader,
                           std::vector<double> &harmonicNumbers,
                           std::vector<double> &rfVoltages, double aatom,
                           double sigs) {
  // output map

  // renaming some of the variables
  double gamma = twissheader["GAMMA"];
  double gammatr = twissheader["GAMMATR"];
  double p0 = twissheader["PC"];
  double len = twissheader["LENGTH"];
  double mass = twissheader["MASS"];
  double charge = twissheader["CHARGE"];
  double *harmonicArr = &harmonicNumbers[0];
  double *voltagesArr = &rfVoltages[0];

  twissheader["sigs"] = sigs;
  twissheader["betar"] = BetaRelativisticFromGamma(gamma);
  twissheader["trev"] = len / (twissheader["betar"] * clight);
  twissheader["frev"] = 1.0 / twissheader["trev"];
  twissheader["omega"] = 2.0 * pi * twissheader["frev"];
  twissheader["eta"] = eta(gamma, gammatr);

  // synchronuous phase
  twissheader["phis"] = SynchronuousPhase(0.0, 173.0, twissheader["U0"], charge,
                                          harmonicNumbers.size(), harmonicArr,
                                          voltagesArr, 1.0e-6);

  // total effective voltage - with possible multi-rf - in electron volt
  twissheader["voltage"] = EffectiveRFVoltageInElectronVolt(
      twissheader["phis"], charge, harmonicNumbers.size(), harmonicArr,
      voltagesArr);

  // synchrotron tune
  twissheader["qs"] = SynchrotronTune(
      twissheader["omega"], twissheader["U0"], charge, harmonicNumbers.size(),
      harmonicArr, voltagesArr, twissheader["phis"], twissheader["eta"], p0);

  // calculate sige and delta
  twissheader["sige"] = sigefromsigs(twissheader["omega"], sigs,
                                     twissheader["qs"], gamma, gammatr);
  twissheader["delta"] = dee_to_dpp(twissheader["sige"], twissheader["betar"]);
}

/*
================================================================================
================================================================================
METHOD TO CALCULATE TIME COEFFICIENT.
================================================================================
AUTHORS:
    -  TOM MERTENS

HISTORY:
    - initial version 02/08/2021

REF:
    - based on original CTE code (CERN) - but updated for multi-RF systems
================================================================================
Arguments:
----------
    - std::map<std::string, double> &twissheaderL
        twissheader map updated with longitudinal parameters
        ( ref: updateTwissHeaderLong )
    - double baseHarmonicNumber
        the base harmonic number of the main RF freq (defining the bucket
        number)

Returns:
--------
    - double tcoeff

================================================================================
================================================================================
*/
double tcoeff(std::map<std::string, double> &twissheaderL,
              double baseHarmonicNumber) {
  return twissheaderL["omega"] * twissheaderL["eta"] * baseHarmonicNumber;
}

/*
================================================================================
================================================================================
METHOD TO CALCULATE MOMENTUM COEFFICIENT.
================================================================================
AUTHORS:
    -  TOM MERTENS

HISTORY:
    - initial version 02/08/2021

REF:
    - based on original CTE code (CERN) - but updated for multi-RF systems
================================================================================
Arguments:
----------
    - std::map<std::string, double> &twissheaderL
        twissheader map updated with longitudinal parameters
        ( ref: updateTwissHeaderLong )

Returns:
--------
    - double pcoeff

================================================================================
================================================================================
*/
double pcoeff(std::map<std::string, double> &twissheaderL, double voltage) {
  return twissheaderL["omega"] * voltage * twissheaderL["CHARGE"] /
         (2.0 * pi * twissheaderL["PC"] * twissheaderL["betar"]);
}

/*
================================================================================
================================================================================
METHOD TO CALCULATE (APPROX) LONGITUDINAL HAMILTONIAN IN FUNCTION OF GIVEN t
(for synchronuous particle t=0).
================================================================================
AUTHORS:
    -  TOM MERTENS

HISTORY:
    - initial version 02/08/2021

REF:
    - based on original CTE code (CERN) - but updated for multi-RF systems
================================================================================
Arguments:
----------
    - std::map<std::string, double> &twissheaderL
        twissheader map updated with longitudinal parameters
        ( ref: updateTwissHeaderLong )
    - std::vector<double> &harmonicNumbers

Returns:
--------
    - double pcoeff

================================================================================
================================================================================
*/
double Hamiltonian(std::map<std::string, double> &twissheaderL,
                   std::vector<double> &harmonicNumbers,
                   std::vector<double> &rfVoltages, double tcoeff, double t) {
  double kinetic, potential;

  // kinetic contribution
  // We assume initial bunch length is given
  kinetic = 0.5 * tcoeff * twissheaderL["delta"] * twissheaderL["delta"];

  std::vector<double> pcoeffs, hRatios, hRatiosInv, phases;

  // calculate coefficients for the determining the potential
  for (int i = 0; i < harmonicNumbers.size(); i++) {
    pcoeffs.push_back(pcoeff(twissheaderL, rfVoltages[i]));
    phases.push_back(harmonicNumbers[i] * twissheaderL["omega"] * t);
    hRatios.push_back(harmonicNumbers[0] / harmonicNumbers[i]);
    hRatiosInv.push_back(harmonicNumbers[i] / harmonicNumbers[0]);
  }

  // calc the potential
  potential = pcoeffs[0] *
              (cos(phases[0]) - cos(twissheaderL["phis"]) +
               (phases[0] - twissheaderL["phis"]) * sin(twissheaderL["phis"]));

  for (int i = 1; i < harmonicNumbers.size(); i++) {
    potential += pcoeffs[i] * hRatios[i] *
                 (cos(phases[i]) - cos(hRatiosInv[i] * twissheaderL["phis"]) +
                  (phases[i] - hRatiosInv[i] * twissheaderL["phis"]) *
                      sin(hRatiosInv[i] * twissheaderL["phis"]));
  }

  return kinetic + potential;
}

} // namespace ste_longitudinal