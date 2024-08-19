# %%
import torch
from numpy import loadtxt,sqrt,ceil
from matplotlib import pyplot as plt
from basis import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, snr = int(sys.argv[1]),float(sys.argv[2])
print(N,10/snr)
shape,scale=10/snr,1

# %%
torch.manual_seed(42)
expo_scale = 1
x = torch.tensor(loadtxt('datasets/E_{}_G_{}_{}.csv'.format(expo_scale,shape,scale))[:N]).float().to(device)
x=x.to(device)
lx = torch.log(x)

a_distribution= log_distribution(torch.distributions.Exponential(torch.tensor(expo_scale*1.0)))

# %%
mu_a,sig_a = a_distribution.mean,a_distribution.stddev
lb_tensor_hor =  torch.linspace(lx.min().item()-mu_a-3*sig_a, lx.max().item()-mu_a+3*sig_a,20000).to(device).reshape(-1,1)
lb_tensor_ver = (1.0*lb_tensor_hor).reshape(-1,1)
dlb = (lb_tensor_hor[1]-lb_tensor_hor[0])

def log_likelihood(ldata,shape,scale):
    lpb = logprob_lgamma(lb_tensor_ver,shape,scale)
    lpa = a_distribution.log_prob(ldata-lb_tensor_ver)   
    lp = lpb+lpa
    return torch.log(dlb)+torch.logsumexp(lp,axis=0)

def logprior(lshape,lscale):
    return logprob_gaussian(lshape,torch.tensor(0),torch.tensor(100))+logprob_gaussian(lscale,torch.tensor(0),torch.tensor(100))


# %%
th_gt = torch.tensor((shape,scale)).float().to(device)
print('gt', log_likelihood(lx,*th_gt).sum())


# %%
sh,sp = torch.meshgrid(torch.linspace(1e-3,5,26),torch.linspace(1e-3,3,26))
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
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-5,ceil(torch.exp(lx.max()).item()),10000).to(device)

p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape).to(device),torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')

i = torch.argmax(torch.stack(lps))
p_map = torch.exp(logprob_gamma(xb,*thetas[i]))
ax[0].plot(xb.cpu(),p_map.cpu(),label='MAP  KL={:.2E}'.format(KL(p_gt,p_map,xb)))

lp = torch.logsumexp(torch.vstack([logprob_gamma(xb,*th) for th in thetas]),axis=0)-torch.log(torch.tensor(len(thetas))).to(device)
p = torch.exp(lp)

ax[0].plot(xb.cpu(),p.cpu(),label='Bayesian reconstruction KL={:.2E}'.format(KL(p_gt,p,xb)))


ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/Prod_Bayes_N_{}_E_{}_G_{}_{}.png'.format(N,expo_scale,10/snr,1),dpi=600)
# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=['shape','scale'])
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/Prod_Bayes_N_{}_E_{}_G_{}_{}.csv'.format(N,expo_scale,10/snr,1))

# %%
with open('report_prod.csv', 'a') as file:
    file.write('E_{}_G_{}_{}.csv'.format(expo_scale,10/snr,1)+',bayes,'+str(N)+','+str(KL(p_gt,p,xb).item())+','+str(KL(p_gt,p_map,xb).item())+'\n')

# %%



