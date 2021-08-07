#include <ibs>
#include <ste>

bool test_updateTwissHeaderLong(map<string, double> &twissheadermap) {
  /*
  uncomment for debuh
  for (auto const &pair : twissheadermap) {
    std::printf("%24s : %16.8e\n", pair.first.c_str(), pair.second);
  }
  */
  if ((twissheadermap["U0"] - 174e3 < 1e-6) &
      (twissheadermap["phis"] - 1.73338693e+02 < 1e-6) &
      (twissheadermap["delta"] - 1.14304713e-03 < 1e-6)) {
    return true;
  }
  return false;
}

bool test_tcoeff(map<string, double> &twissheadermap) {
  double baseh = 400.0;

  /* uncomment to debug
  std::printf("tcoeff : %20.15e\n",
              ste_longitudinal::tcoeff(twissheadermap, baseh));
  */
  if (ste_longitudinal::tcoeff(twissheadermap, baseh) - 2.297095009748885e+06 <
      1e-6) {
    return true;
  }
  return false;
}

bool test_pcoeff(map<string, double> &twissheadermap) {
  double h = 400.0;
  double v = -1.5e6;
  double tc = 2.3e6;
  double t = 0.0;

  double value = ste_longitudinal::pcoeff(twissheadermap, v);
  /* uncomment to debug
  std::printf("%24.15e\n", value);
  */
  if (value - 1.102178204333954e+03 < 1e-6) {
    return true;
  }
  return false;
}

bool test_hamiltonian(map<string, double> &twissheadermap,
                      std::vector<double> &harmonicNumbers,
                      std::vector<double> &rfVoltages) {
  double tc = 2.297095009748885e+06;
  double t = 0.0;
  double delta = twissheadermap["delta"];

  double value = ste_longitudinal::Hamiltonian(twissheadermap, harmonicNumbers,
                                               rfVoltages, tc, t, delta);
  /* uncomment to debug
  std::printf("%24.16e\n", value);
  */

  if (value - 1.0208078672494565e+05 < 1.0e-6) {
    return true;
  }

  return false;
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

  // test update twissheader longitudinal
  std::printf("%-30s", "test_updateTwissHeaderLong");
  if (test_updateTwissHeaderLong(twissheadermap)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test tcoeff
  std::printf("%-30s", "test_tcoeff");
  if (test_tcoeff(twissheadermap)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test pcoeff
  std::printf("%-30s", "test_pcoeff");
  if (test_pcoeff(twissheadermap)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // test hamiltonian
  std::printf("%-30s", "test_hamiltonian");
  if (test_hamiltonian(twissheadermap, h, v)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  int nMacro = 2;
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

  /*
   std::transform(
       distribution.begin(), distribution.end(), distribution.begin(),
       ste_longitudinal::RfUpdateRoutineFunctor(1.0, twissheadermap, h, v));
       */
  return 0;
}