import IBSLib as ibslib
import numpy as np
import STELib as stelib
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

twheader = ibslib.GetTwissHeader(
    "/home/mti/github-tomerten/steibs/cpp/tests/src/b2_design_lattice_1996.twiss"
)
twiss = ibslib.GetTwissTable(
    "/home/mti/github-tomerten/steibs/cpp/tests/src/b2_design_lattice_1996.twiss"
)
twheader["U0"] = 174000
twheader = stelib.updateTwissHeaderLong(twheader, h, v, aatom, sigs)

betxavg = twheader["LENGTH"] / (twheader["Q1"] * 2 * np.pi)
betyavg = twheader["LENGTH"] / (twheader["Q2"] * 2 * np.pi)
stelib.BiGaussian6D(betxavg, ex, betyavg, ey, h, v, twheader, seed)

for _ in range(2 ** 11):
    #    test4.append(np.array(stelib.BiGaussian4D(betxavg,ex,betyavg,ey,seed)))
    test6.append(
        np.array(stelib.BiGaussian6DLongMatched(betxavg, ex, betyavg, ey, h, v, twheader, seed))
    )
    # test6.append(np.array(stelib.BiGaussian6D(betxavg,ex,betyavg,ey,h,v,twheader,seed)))
# test4 = np.array(test4)
test6m = np.array(test6)

# n = 2 ** 0
# twheader["phis"] = twheader["phis"] + 360.0
# test6m = np.array(stelib.GenerateDistribution(n, betxavg, ex, betyavg, ey, h, v, twheader, seed))
# test6m = np.array(stelib.GenerateDistributionMatched(n,betxavg,ex,betyavg,ey,h,v,twheader,seed))

import ste

# ste.distributionplot(test6m)

twheader["timeratio"] = 2000
twheader["aatom"] = ibslib.electron_mass / ibslib.proton_mass
equi = stelib.radiationEquilib(twheader)
print(twheader)
print(equi)
# print(test6m[0])
# test6m = np.array(stelib.betatronUpdate(test6m, twheader, equi, seed))
# print(test6m[0])
# fig, axs = plt.subplots(2,2,figsize=(15,15))

# axs[0,0].scatter(test6[:,4],test6[:,5],s=1)
# axs[0,1].scatter(test6m[:,4],test6m[:,5],s=1)
# axs[1,0].hist(test6[:,4],bins=20)
# axs[1,1].hist(test6m[:,4],bins=20)

# plt.show()
plt.ion()
fig = plt.figure(constrained_layout=True, figsize=(14, 10))

# create plot grid
gs = fig.add_gridspec(2, 4)

# create subplots and set titles
ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[0, 2:])
ax4 = fig.add_subplot(gs[1, 0])
ax5 = fig.add_subplot(gs[1, 1])
ax6 = fig.add_subplot(gs[1, 2])
ax7 = fig.add_subplot(gs[1, 3])

ax1.set_title(r"$x-p_x$ init")
ax2.set_title(r"$y-p_y$ init")
ax3.set_title(r"$t-d\gamma$ init")
ax4.set_title(r"L ip1")
ax5.set_title(r"L ip2")
ax6.set_title(r"L ip5")
ax7.set_title(r"L ip8")

K2L = np.sum(twiss["K2L"])
K2SL = np.sum(twiss["K2SL"])
for turn in range(0, 100):
    ax1.scatter(test6m[:, 0], test6m[:, 1], s=1)
    ax1.set_title("Turn {}".format(turn))
    ax1.set_xlim(-0.001, 0.001)
    ax2.set_xlim(-0.001, 0.001)
    ax3.set_xlim(-5e-9, 5e-9)
    ax1.set_ylim(-0.001, 0.001)
    ax2.set_ylim(-0.001, 0.001)
    ax3.set_ylim(-3, 3)
    ax4.set_xlim(0, 100)
    ax5.set_xlim(0, 100)
    ax6.set_xlim(0, 100)
    ax7.set_xlim(0, 100)
    plt.pause(0.001)
    ax1.cla()
    ax2.cla()
    ax3.cla()

    test6m = np.array(stelib.betatronUpdate(test6m, twheader, 0.0, 0, 0))
    # plt.show(block=True)
plt.ioff()
