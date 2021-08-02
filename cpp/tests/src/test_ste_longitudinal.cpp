#include <ibs>
#include <ste>

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

  for (auto const &pair : twissheadermap) {
    std::printf("%24s : %16.8e\n", pair.first.c_str(), pair.second);
  }
  return 0;
}