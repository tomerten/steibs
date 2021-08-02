#include <ibs>
#include <ste>

int main() {
  std::string twissfilename = "../src/b2_design_lattice_1996.twiss";
  std::map<std::string, double> twissheadermap;
  twissheadermap = GetTwissHeader(twissfilename);

  std::printf("%12.5e", twissheadermap["GAMMA"]);
  return 0;
}