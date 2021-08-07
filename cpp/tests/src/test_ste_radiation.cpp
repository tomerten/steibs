#include <ibs>
#include <iomanip>
#include <iostream>
#include <ste>

// for testing
template <typename T>
bool RequireVectorEquals(const std::vector<T> &a, const std::vector<T> &b,
                         T max_error = 0.000000001) {
  const auto first_mismatch = std::mismatch(
      a.begin(), a.end(), b.begin(), [max_error](T val_a, T val_b) {
        // std::cout << val_a << " " << val_b << std::endl;
        return val_a == val_b || std::abs(val_a - val_b) < max_error;
      });
  if (first_mismatch.first != a.end()) {
    // std::printf("%s", std::to_string(*first_mismatch.first).c_str());
    // std::printf("%s", std::to_string(*first_mismatch.second).c_str());
    return false;
  } else
    return true;
}

bool test_rad_losses(std::map<std::string, double> &twiss) {
  double expected = 1.696758e+05;
  double actual = ste_radiation::RadiationLossesPerTurn(twiss);
  if (abs(actual - expected) < 1.0) {
    return true;
  }
  return false;
}

bool test_rad_equi(std::map<std::string, double> &twiss) {
  std::vector<double> expected = {
      5.12638540e-09, 9.41556346e-14, 1.00322520e+00,
      1.00000000e+00, 4.87765480e-07, 3.04467213e-03,
      4.00396162e-03, 7.99505120e-03, 8.02083681e-03};

  std::map<std::string, double> equi;
  std::vector<double> actual;
  equi = ste_radiation::radiationEquilib(twiss);

  for (map<string, double>::const_iterator it = equi.begin(); it != equi.end();
       ++it) {
    actual.push_back(it->second);
  }
  return RequireVectorEquals(expected, actual, 0.1);
}

int main() {
  // set seed
  int seed = 123456;

  string twissfilename = "../src/b2_design_lattice_1996.twiss";
  map<string, double> twissheadermap;
  twissheadermap = GetTwissHeader(twissfilename);

  map<string, std::vector<double>> twiss = GetTwissTableAsMap(twissfilename);
  // rf settings
  std::vector<double> h, v;
  h.push_back(400.0);
  v.push_back(-1.5e6);

  // bunch length
  double sigs = 0.005;

  // aatom
  double aatom = emass / pmass;

  // first add aatom to twheader
  twissheadermap["aatom"] = emass / pmass;

  // updateTwiss with rad
  updateTwiss(twiss);

  // update twiss header with long parameters
  ste_longitudinal::updateTwissHeaderLong(twissheadermap, h, v, aatom, sigs);

  // test radiation losses per turn
  std::printf("%-30s", "test_rad_losses ");
  if (test_rad_losses(twissheadermap)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();

  // set energy loss per turn
  twissheadermap["U0"] = ste_radiation::RadiationLossesPerTurn(twissheadermap);

  // compare rad integrals
  std::map<std::string, double> radint =
      ste_radiation::radiationIntegrals(twiss);
  std::map<std::string, std::string> madmap;
  madmap["I2"] = "SYNCH_2";
  madmap["I3"] = "SYNCH_3";
  madmap["I4x"] = "SYNCH_4";
  madmap["I4y"] = "SYNCH_4";
  madmap["I5x"] = "SYNCH_5";
  madmap["I5y"] = "SYNCH_5";

  /*
  for (map<string, double>::const_iterator it = radint.begin();
       it != radint.end(); ++it) {
    std::printf("%-20s %16.8e %16.8e \n", it->first.c_str(), it->second,
                twissheadermap[madmap[it->first]]);
  }
  std::map<std::string, double> equi;
  equi = ste_radiation::radiationEquilib(twissheadermap);

  for (map<string, double>::const_iterator it = equi.begin(); it != equi.end();
       ++it) {
    std::printf("%-20s %16.8e\n", it->first.c_str(), it->second);
  }
*/
  // test radiation losses per turn
  std::printf("%-30s", "test_rad_equi ");
  if (test_rad_equi(twissheadermap)) {
    ste_output::green();
    std::printf("Passed\n");
  } else {
    ste_output::red();
    std::printf("Failed\n");
  }
  ste_output::reset();
  return 0;
}