# %%
import torch
from numpy import loadtxt,sqrt,ceil
from matplotlib import pyplot as plt
from basis import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, shape = int(sys.argv[1]),int(sys.argv[2])
print(N,shape)

# %%
torch.manual_seed(42)
mu_a,sig_a = 10,1
shape,scale = shape,1
filename = 'prod_N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float().to(device)
lx = torch.log(x)

a_distribution= torch.distributions.Normal(torch.tensor(mu_a).float().to(device),torch.tensor(sig_a).float().to(device))
la_distribution = log_distribution(a_distribution)

# %%
la_mean,la_sig = la_distribution.mean,la_distribution.stddev
lb_tensor_hor =  torch.linspace(lx.min().item()-la_mean-3*sig_a, lx.max().item()-la_sig+3*sig_a,20000).to(device).reshape(-1,1)
lb_tensor_ver = (1.0*lb_tensor_hor).reshape(-1,1)
dlb = (lb_tensor_hor[1]-lb_tensor_hor[0])

def log_likelihood(ldata,shape,scale):
    lpb = logprob_lgamma(lb_tensor_ver,shape,scale)
    lpa = la_distribution.log_prob(ldata-lb_tensor_ver)   
    lp = lpb+lpa
    return torch.log(dlb)+torch.logsumexp(lp,axis=0)

def logprior(lshape,lscale):
    return logprob_gaussian(lshape,torch.tensor(0),torch.tensor(100))+logprob_gaussian(lscale,torch.tensor(0),torch.tensor(100))


# %%
th_gt = torch.tensor((shape,scale)).float().to(device)
print('gt', log_likelihood(lx,*th_gt).sum())


# %%
sh,sp = torch.meshgrid(torch.linspace(1e-3,10,26),torch.linspace(1e-3,6,26))
sh,sp = sh.reshape(-1).to(device), sp.reshape(-1).to(device)
lps = [log_likelihood(lx,shi,spi).sum() for shi,spi in zip(sh,sp)]
i = torch.argmax(torch.stack(lps))
th = torch.stack((sh[i],sp[i]))
lth = torch.log(th)
lp = lps[i] + logprior(*lth)

# %%
lp_function = lambda lth: log_likelihood(lx,*torch.exp(lth)).sum() + logprior(*lth)

def prop(lth,tol=1e-2):
    return lth+tol*torch.randn(2).to(device)

def update(lth,lp,tol=1e-2):
    lth_prop = prop(lth,tol)
    lp_prop = lp_function(lth_prop)
    if torch.log(torch.rand(1))<(lp_prop-lp).item():
        lth = lth_prop
        lp = lp_prop
    return lth,lp

# %%
thetas = []
lps = []

for i in range(10000):
    lth, lp = update(lth,lp,tol=1e-2)

    thetas.append(torch.exp(lth))
    lps.append(lp)
    if (i-1)%1000==0:
        print(i,thetas[-1],lp)


# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=['shape','scale'])
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/prod_bayes_'+filename.split('.')[0]+'datapoints_{}.csv'.format(N))




