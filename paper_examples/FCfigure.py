# %%
import pandas as pd
import nf_class
import torch
from matplotlib import pyplot as plt

import numpy as np
from matplotlib.ticker import ScalarFormatter
import matplotlib.gridspec as gridspec

from tqdm import tqdm
torch.manual_seed(0)

# %%
device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

control = pd.read_csv('FCdata/complete_d=0.33.csv')['FL1-A'].to_numpy()
control = torch.tensor(control).to(device).float()

a_dist = nf_class.NormalizingFlow_shifted(device=device,layers=2)
a_dist.load_state_dict(torch.load('models/FCdata_control_NF.pt',weights_only=False))

x_ctr = torch.linspace(control.min(),control.max(),101,device=device)
p_ctr = torch.exp(a_dist.log_prob(x_ctr.reshape(-1,1)).reshape(-1))

# %%
lowstress = pd.read_csv('FCdata/complete_d=0.23.csv')['FL1-A'].to_numpy()
lowstress = torch.tensor(lowstress).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
lowstress_deconvolver = nf_class.Deconvolver(lowstress[:10000],torch.distributions.Normal(0,1),intervals=1000)
lowstress_deconvolver.load_state_dict(torch.load('models/FCdata_lowstress_NF.pt'))

# %%
highstress = pd.read_csv('FCdata/complete_d=0.12.csv')['FL1-A'].to_numpy()
highstress = torch.tensor(highstress).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()
highstress_deconvolver = nf_class.Deconvolver(highstress[:10000],torch.distributions.Normal(0,1),intervals=1000)
highstress_deconvolver.load_state_dict(torch.load('models/FCdata_highstress_NF.pt'))

# %%
lx = torch.linspace(-1,13,101)

#Since there are many values close to 0, and they were incorrectly interpreted, I am moving towards this taking the log of the histogram approach

low_hist = torch.log(lowstress_deconvolver.nfs.sample((100000))[0].reshape(-1).clamp(1)).detach().cpu()
low_density, bin_edges = np.histogram(low_hist, bins=lx, density=True)
low_bin_centers = (bin_edges[:-1] + bin_edges[1:])/2
del low_hist

high_hist = torch.log(highstress_deconvolver.nfs.sample((100000))[0].reshape(-1).clamp(1)).detach().cpu()
high_density, bin_edges = np.histogram(high_hist, bins=lx, density=True)
high_bin_centers = (bin_edges[:-1] + bin_edges[1:])/2


# %%

fig = plt.figure(figsize=(13.3, 6))
gs = gridspec.GridSpec(4, 4, height_ratios=[1, 0.33, 1, 1])

ax = np.empty((3,4), dtype=object)
for row_idx, gs_row in enumerate([0,2,3]):
    for col_idx in range(4):
        ax[row_idx, col_idx] = fig.add_subplot(gs[gs_row, col_idx])

ax[0,0].hist(control.cpu(),density=True,bins=50)
ax[0,0].plot(x_ctr.cpu(),p_ctr.detach().cpu())
ax[0,1].hist(torch.log(control.cpu()),density=True,bins=50)
ax[0,1].plot(torch.log(x_ctr.cpu()),(p_ctr*x_ctr).detach().cpu())


x, lp = [x.cpu() for x in lowstress_deconvolver.get_pdf(log_prob=True)]
ax[1,0].hist(lowstress.cpu(),density=True,bins=50)
ax[1,1].hist(torch.log(lowstress.cpu()),density=True,bins=50)
ax[1,2].plot(x,torch.exp(lp),color='r')
ax[1,3].plot(low_bin_centers,low_density,color='r')

x, lp = [x.cpu() for x in highstress_deconvolver.get_pdf(log_prob=True)]
ax[2,0].hist(highstress.cpu(),density=True,bins=50)
ax[2,1].hist(torch.log(highstress.cpu()),density=True,bins=50)
ax[2,2].plot(x,torch.exp(lp),color='r')
ax[2,3].plot(high_bin_centers,high_density,color='r')

[(axi.set_xlim(5.5,12.5),axi.set_ylim(0,1.1)) for axi in ax[:,1]]

[(axi.xaxis.set_major_formatter(ScalarFormatter(useMathText=True)),
  axi.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))) for axi in ax.T[[0,2],:].reshape(-1)]
[axi.set_xlim(-1e4,1.5*1e5) for axi in ax[:,0]]
[(axi.yaxis.set_major_formatter(ScalarFormatter(useMathText=True)),
  axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))) for axi in ax.reshape(-1)]

ax[0,0].set_ylabel('Control',fontsize=14)
[axi.set_xlabel(r'$a$',fontsize=16) for axi in ax[0,:]]

ax[1,0].set_ylabel('Low stress',fontsize=14)
[axi.set_xlabel(r'$x$',fontsize=16) for axi in ax[1:3, 0:2].reshape(-1)]
ax[2,0].set_ylabel('High stress',fontsize=14)
[axi.set_xlabel(r'$b$',fontsize=16) for axi in ax[1:3, 2:4].reshape(-1)]
[axi.set_xlabel(r'$\log$'+ axi.get_xlabel(),fontsize=16) for axi in ax[:,[3,1]].reshape(-1)]
[(axi.set_xticklabels([]),axi.set_xlabel('')) for axi in ax[1]] #Sharex at home

fig.text(0.3, 0.92, 'Autofluorescence intensity (AFU)', ha='center', fontsize=16)
fig.text(0.3, 0.55, '     Total intensity (AFU)      ', ha='center', fontsize=16)
fig.text(0.7, 0.55, '  NFdeconvolve reconstruction   ', ha='center', fontsize=16)
[ax[0, i].axis('off') for i in range(2,4)]

gs.update(hspace=0.3)
plt.savefig('graphs/NF_result.png',dpi=600)

# %%



