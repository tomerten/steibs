import pandas as pd
from matplotlib import pyplot as plt
from matplotlib  import rcParams

df= pd.read_csv("rands_bigaussian_out.txt",delim_whitespace=True,header=None)
df2 = pd.read_csv("rands_6D_bigauss0_out.txt",delim_whitespace=True,header=None)
df3 = pd.read_csv("rands_6D_bigauss1_out.txt",delim_whitespace=True,header=None)
df4 = pd.read_csv("rands_6D_bigauss0_matched_out.txt",delim_whitespace=True,header=None)
df5 = pd.read_csv("rands_6D_bigauss1_matched_out.txt",delim_whitespace=True,header=None)
df6 = pd.read_csv("rands_6D_bigauss0_matched_out_rfupdate.txt",delim_whitespace=True,header=None)
df7 = pd.read_csv("rands_6D_bigauss0_matched_out_rfupdate_raddamp.txt",delim_whitespace=True,header=None)
df8 = pd.read_csv("rands_6D_bigauss0_matched_out_rfupdate_raddamp_betatron.txt",delim_whitespace=True,header=None)
df9 = pd.read_csv("rands_6D_bigauss0_matched_out_rfupdate_raddamp_betatron_ibs.txt",delim_whitespace=True,header=None)

fig, axes = plt.subplots(nrows=2,ncols=2)

rcParams['figure.figsize'] = 20, 30
fig.set_size_inches(18.5, 10.5, forward=True)

axes[0,0].scatter(df[0],df[1],s=0.2)

#axes[0,1].scatter(df2[0],df2[1],s=0.2)
# axes[0,1].scatter(df3[0],df3[1],s=0.2)
axes[0,1].scatter(df4[0],df4[1],s=0.3)
#axes[0,1].scatter(df6[0],df6[1],s=0.3)
#axes[0,1].scatter(df7[0],df7[1],s=0.3)
axes[0,1].scatter(df8[0],df8[1],s=0.3)
axes[0,1].scatter(df9[0],df9[1],s=0.3)

#axes[1,0].scatter(df2[2],df2[3],s=0.2)
# axes[1,0].scatter(df3[2],df3[3],s=0.2)
axes[1,0].scatter(df4[2],df4[3],s=0.3)
#axes[1,0].scatter(df6[2],df6[3],s=0.3)
#axes[1,0].scatter(df7[2],df7[3],s=0.3)
axes[1,0].scatter(df8[2],df8[3],s=0.3)
axes[1,0].scatter(df9[2],df9[3],s=0.3)

#axes[1,1].scatter(df2[4],df2[5],s=0.2)
# axes[1,1].scatter(df3[4],df3[5],s=0.2)
axes[1,1].scatter(df4[4],df4[5],s=0.2)
#axes[1,1].scatter(df5[4],df5[5],s=0.2)
#axes[1,1].scatter(df6[4],df6[5],s=0.3)
#axes[1,1].scatter(df7[4],df7[5],s=0.3)
axes[1,1].scatter(df8[4],df8[5],s=0.3)
axes[1,1].scatter(df9[4],df9[5],s=0.3)


axes[1,1].set_xlim(-2.2e-9,0.25e-9)
plt.show()
