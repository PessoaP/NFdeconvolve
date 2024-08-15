# %%
import torch
from numpy import loadtxt,sqrt,ceil
from matplotlib import pyplot as plt
from basis import *
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, sig = int(sys.argv[1]),float(sys.argv[2])
if sig ==int(sig):
    sig=int(sig)
print(N,sig)

# %%
torch.manual_seed(42)
mu_a,sig_a = 100,sig
scale = 1
x = torch.tensor(loadtxt('datasets/N_{}_{}_E_{}.csv'.format(mu_a,sig_a,scale))[:N]).float().to(device)

x=x.to(device)

# %%
b_tensor_hor = torch.linspace(1e-3,ceil(x.max().item()/mu_a),10000).reshape(-1).to(device)
b_tensor_ver = (1.0*b_tensor_hor).reshape(-1,1)
db = (b_tensor_hor[1]-b_tensor_hor[0])

def log_likelihood(data,scale,mu=torch.tensor(mu_a).to(device),sig=torch.tensor(sig_a).to(device),N_samples=10000):
    lpb = logprob_exp(b_tensor_ver,scale)
    lpa = logprob_gaussian(data/b_tensor_ver,mu,sig)
    
    lp = lpb+lpa-torch.log(torch.abs(b_tensor_ver))
    #return torch.log(db*torch.sum(torch.exp(lpb+lpa),axis=0))
    return torch.log(db)+torch.logsumexp(lp,axis=0)


def logprior(lscale):
    return logprob_gaussian(lscale,torch.tensor(0),torch.tensor(100))

# %%
th_gt = torch.tensor((scale)).float().to(device)
log_likelihood(x,th_gt).sum()+ logprior(torch.log(th_gt))


# %%
sh = torch.linspace(1e-3,10,99).to(device)
lps = [log_likelihood(x,shi).sum() for shi in sh]
i = torch.argmax(torch.stack(lps))
th = sh[i]
lth = torch.log(th)
lp = lps[i] + logprior(lth)

# %%
lp_function = lambda lth: log_likelihood(x,torch.exp(lth)).sum() + logprior(lth)

def prop(lth,tol=1e-2):
    return lth+tol*torch.randn(1).to(device)

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

for i in range(20000):
    lth, lp = update(lth,lp,tol=1e-2)

    thetas.append(torch.exp(lth))
    lps.append(lp)
    if (i-1)%1000==0:
        print(i,thetas[-1],lp)

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-3,ceil(x.max().item()/mu_a),2001).to(device)

p_gt = torch.exp(logprob_exp(xb,torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')


i = torch.argmax(torch.stack(lps))
p_map = torch.exp(logprob_exp(xb,thetas[i]))
ax[0].plot(xb.cpu(),p_map.cpu(),label='MAP  KL={:.2E}'.format(KL(p_gt,p_map,xb)))

p = torch.exp(torch.vstack([logprob_exp(xb,th) for th in thetas])).mean(axis=0)
ax[0].plot(xb.cpu(),p.cpu(),label='Bayesian reconstruction KL={:.2E}'.format(KL(p_gt,p,xb)))


ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/Bayes_prod_N_{}_{}_E_{}_datapoints{}.png'.format(mu_a,sig_a,scale,N),dpi=600)

# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=['scale'])
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/prod_bayes_N_{}_{}_E_{}_datapoints{}.csv'.format(mu_a,sig_a,scale,N))

# %%
with open('report_prod.csv', 'a') as file:
    file.write('N_{}_{}_E_{}.csv'.format(mu_a,sig_a,scale)+',bayes,'+str(N)+','+str(KL(p_gt,p,xb).item())+','+str(KL(p_gt,p_map,xb).item())+'\n')

# %%



