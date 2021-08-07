import STELib as stelib
import IBSLib as ibslib
import numpy as np
from matplotlib import pyplot as plt

test6 = []
test4 = []

seed = 123456
coupling = 0.05
ex = 5e-9
ey = coupling * ex
h = [400.0]
v = [-1.5e6]
aatom = ibslib.electron_mass / ibslib.proton_mass
sigs = 0.005

twheader = ibslib.GetTwissHeader("/home/mti/github-tomerten/steibs/cpp/tests/src/b2_design_lattice_1996.twiss")
twheader["U0"] = 174000
twheader = stelib.updateTwissHeaderLong(twheader,h,v,aatom,sigs)

betxavg = twheader["LENGTH"] /(twheader["Q1"] * 2*np.pi )
betyavg = twheader["LENGTH"] /(twheader["Q2"] * 2*np.pi )
stelib.BiGaussian6D(betxavg,ex,betyavg,ey,h,v,twheader,seed)

for _ in range(2**11):
#    test4.append(np.array(stelib.BiGaussian4D(betxavg,ex,betyavg,ey,seed)))
    test6.append(np.array(stelib.BiGaussian6DLongMatched(betxavg,ex,betyavg,ey,h,v,twheader,seed)))
    #test6.append(np.array(stelib.BiGaussian6D(betxavg,ex,betyavg,ey,h,v,twheader,seed)))
#test4 = np.array(test4)
test6m = np.array(test6)

n = 2**11
#test6m = np.array(stelib.GenerateDistribution(n,betxavg,ex,betyavg,ey,h,v,twheader,seed))
#test6m = np.array(stelib.GenerateDistributionMatched(n,betxavg,ex,betyavg,ey,h,v,twheader,seed))

import ste

ste.distributionplot(test6m)

#fig, axs = plt.subplots(2,2,figsize=(15,15))

#axs[0,0].scatter(test6[:,4],test6[:,5],s=1)
#axs[0,1].scatter(test6m[:,4],test6m[:,5],s=1)
#axs[1,0].hist(test6[:,4],bins=20)
#axs[1,1].hist(test6m[:,4],bins=20)

#plt.show()
