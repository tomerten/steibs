#include <ibs>
#include <numeric>
#include <ste>

int main() {
  // set seed
  int seed = 123456;
  string twissfilename = "../src/b2_design_lattice_1996.twiss";
  map<string, double> twissheadermap;
  twissheadermap = GetTwissHeader(twissfilename);
  std::map<std::string, std::vector<double>> twiss =
      GetTwissTableAsMap(twissfilename);

  // rf settings
  std::vector<double> h, v;
  h.push_back(400.0);
  v.push_back(-1.5e6);

  // bunch length
  double sigs = 0.005;

  // aatom
  double aatom = emass / pmass;
  twissheadermap["aatom"] = emass / pmass;
  // set energy loss per turn manually
  twissheadermap["U0"] = ste_radiation::RadiationLossesPerTurn(twissheadermap);

  // update twiss header with long parameters
  ste_longitudinal::updateTwissHeaderLong(twissheadermap, h, v, aatom, sigs);

  int nMacro = 2;
  double timeratio = 2000;
  twissheadermap["timeratio"] = timeratio;
  double betx = twissheadermap["LENGTH"] / (twissheadermap["Q1"] * 2.0 * pi);
  double bety = twissheadermap["LENGTH"] / (twissheadermap["Q2"] * 2.0 * pi);
  double coupling = 0.05;
  double ex = 5e-9;
  double ey = coupling * ex;

  std::vector<std::vector<double>> distribution =
      ste_random::GenerateDistributionMatched(nMacro, betx, ex, bety, ey, h, v,
                                              twissheadermap, seed);

  ste_output::printVector(distribution[0]);
  ste_longitudinal::RfUpdate(distribution, 1.0, twissheadermap, h, v);
  ste_output::printVector(distribution[0]);

  std::map<std::string, double> equi;
  std::vector<double> actual;
  equi = ste_radiation::radiationEquilib(twissheadermap);
  ste_radiation::RadUpdate(distribution, twissheadermap, equi, seed);
  ste_output::printVector(distribution[0]);
  ste_radiation::RadUpdate(distribution, twissheadermap, equi, seed);
  ste_output::printVector(distribution[0]);

  double K2L = std::accumulate(twiss["K2L"].begin(), twiss["K2L"].end(), 0.0);
  double K2SL =
      std::accumulate(twiss["K2SL"].begin(), twiss["K2SL"].end(), 0.0);

  std::printf("%-30s %16.8e\n", "K2L", K2L);
  std::printf("%-30s %16.8e\n", "K2SL", K2SL);

  ste_global::betatronUpdate(distribution, twissheadermap, 0.05, K2L, K2SL);
  ste_output::printVector(distribution[0]);
  return 0;
}