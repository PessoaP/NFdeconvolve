# %%
import torch

from basis import *
import numpy as np
import matplotlib.pyplot as plt

import nf_class
import npbayes_class

# %%
np.random.seed(42)
t = lambda x: torch.tensor(x).to(device)
fsize = (5,3)

# %%
print('Gamma/Inverse Gamma Mixture example in Fig. 1')
N_base = 2000

name='gammamix'
def b_gt(x):
    p1 = torch.exp(logprob_gamma(x,t(3/2),t(2))) 
    p2 = torch.exp(logprob_gamma(15-x,t(3/2),t(2))) 
    p1,p2 = torch.nan_to_num(p1,nan=0),torch.nan_to_num(p2,nan=0)
    return .7*p1+.3*p2
b = np.concatenate((np.random.gamma(3/2,2,size=7*N_base),15-np.random.gamma(3/2,2,size=3*N_base)))
a = np.random.normal(size=b.size)
a_dist = torch.distributions.Normal(0,1)
x=a+b

NF = nf_class.Deconvolver(torch.tensor(x),a_dist,intervals=10000)
NP = npbayes_class.Deconvolver(torch.tensor(x),0,1)

# %%
NF.train()
th,lps = NP.MCMC_chain(samples=20000)
map_index = torch.argmax(lps)

# %%
comparative_plot(name,NF,th,b_gt,map_index,x,b,tails=(-3,18),loc=1)
# %%
def fix_axes(ax):
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) 

    ymax = ax.get_ylim()[1]
    pow = np.floor(np.log10(ymax))
    ticks = ax.get_yticks()
        
    if len(ticks) > 3:
        ticks = np.linspace(0, ticks[-1], 3)
        ticks = (ticks/(10**pow)).astype(int)*(10**pow)

    ax.set_yticks(ticks)
# %%
fig,ax = plt.subplots(1,1,figsize=(fsize))
g = plt.hist(x,density=True,bins=49)
plt.title('Data',fontsize=20)
plt.xlim(-3,18)

plt.xlabel(r'$x$',fontsize=15)
plt.ylabel('Density',fontsize=15)

fix_axes(ax)
plt.tight_layout()
plt.savefig('graphs/fig1_data.png',dpi=1000)

# %%
fig,ax = plt.subplots(1,1,figsize=(fsize))
xb = np.linspace(-3,18,101)
plt.hist(b,density=True,bins=g[1],)
plt.plot(xb,b_gt(t(xb)).cpu(),color='k',label = 'Ground Truth',linewidth=2)

plt.title('Signal (Unknown)',fontsize=20)
plt.xlim(-3,18)

plt.xlabel(r'$b$',fontsize=15)
plt.ylabel('Density',fontsize=15)

fix_axes(ax)
plt.tight_layout()
plt.savefig('graphs/fig1_signal.png',dpi=1000)

xb,pnf = NF.get_pdf()
ax.plot(xb.cpu(),pnf.cpu(),color='r',label = 'NFdeconvolve',linewidth=2)
ax.set_title('Signal distribution (Learned)',fontsize=20)

plt.legend(fontsize=15)
fix_axes(ax)
plt.tight_layout()
plt.savefig('graphs/fig1_deconvoluted.png',dpi=1000)

# %%
fig,ax = plt.subplots(1,1,figsize=(fsize))
xa = np.linspace(-3.5,3.5,101)

plt.plot(xa,
         torch.exp(a_dist.log_prob(t(xa)).cpu()),
         color='k',label = 'Distribution',linewidth=2)
plt.title('Noise distribution (known)',fontsize=20)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
plt.xlabel(r'$a$',fontsize=15)
plt.ylabel('Density',fontsize=15)
plt.xlim(-3.45,3.45)
fix_axes(ax)
plt.tight_layout()
plt.savefig('graphs/fig1_noisedist.png',dpi=1000)

plt.title('Noise',fontsize=20)
plt.hist(a,density=True,bins=49,)

plt.tight_layout()
plt.savefig('graphs/fig1_noise.png',dpi=1000)

del NF
del NP
# %%
print('Gaussian Mixture example in Fig. 2')
N_base = 2000
name='normalmix'

b_gt = lambda x: .5*torch.exp(logprob_gaussian(x,t(-5),t(1.5)))+.5*torch.exp(logprob_gaussian(x,t(5),t(1.5)))
b = np.concatenate((np.random.normal(-5,1.5,size=10000),np.random.normal(5,1.5,size=10000)))
a = np.random.normal(0,1,size=20000)
a_dist = torch.distributions.Normal(0,1)
x=a+b

NF = nf_class.Deconvolver(torch.tensor(x),a_dist,intervals=10000,tail_bound=10)
NP = npbayes_class.Deconvolver(torch.tensor(x),0,1) 

# %%
NF.train()
th,lps = NP.MCMC_chain(samples=20000)
map_index = torch.argmax(lps)

comparative_plot(name,NF,th,b_gt,map_index,x,b,tails=(-18,18))