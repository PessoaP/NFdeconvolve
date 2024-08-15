# %%
import torch
from numpy import loadtxt,sqrt,ceil,linspace
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

def logprob_mixlognormal(x,mus,sigs,rhos):
    return torch.logsumexp(logprob_gaussian(torch.log(x),mus.reshape(-1),sigs.reshape(-1))
                           -torch.log(x)
                           +torch.log(rhos.reshape(-1)),axis=1,keepdim=True)


def log_likelihood(data,mus,sigs,rhos,mu=torch.tensor(mu_a).to(device),sig=torch.tensor(sig_a).to(device),N_samples=10000):
    lpb = logprob_mixlognormal(b_tensor_ver,mus,sigs,rhos)
    lpa = logprob_gaussian(data/b_tensor_ver,mu,sig)

    lp = lpb+lpa-torch.log(torch.abs(b_tensor_ver))
    return torch.log(db)+torch.logsumexp(lp,axis=0)


Ncomp = 20

diralpha = ((.9)**torch.arange(Ncomp)).to(device)
diralpha = normalize(diralpha)/10
dir_prior = torch.distributions.Dirichlet(diralpha)

def logprior(mus,sigs,rhos): 
    return (logprob_gaussian(mus,torch.zeros_like(diralpha),5*torch.ones_like(diralpha)).sum() 
            + logprob_gaussian(torch.log(sigs),torch.ones_like(diralpha),1*torch.ones_like(diralpha)).sum() 
            - torch.log(sigs).sum()
            + dir_prior.log_prob(rhos))

lp_function = lambda mus,sigs,rhos: log_likelihood(x,mus,sigs,rhos).sum() + logprior(mus,sigs,rhos)


# %%
# %%
def prop_mus(mus,tol=1e-2):
    return mus+tol*torch.randn(Ncomp).to(device)

def prop_sigs(sigs,tol=1e-2):
    return torch.exp(torch.log(sigs)+tol*torch.randn(Ncomp).to(device))

dir = torch.distributions.Dirichlet(torch.ones(Ncomp).to(device))
def prop_rhos(rhos,dir=dir,tol=1e-2):
    return tol*dir.sample()+(1-tol)*rhos

def log_propdist_rhos(prop,origin,dir=dir,tol=1e-2):
    beta = (prop-(1-tol)*origin)/tol
    return dir.log_prob(beta) + torch.log(torch.tensor(tol))*Ncomp

#dir = torch.distributions.Dirichlet(diralpha)
def prop_rhos(rhos,diralpha=diralpha,tol=1e-5):
    dir = torch.distributions.Dirichlet(rhos/tol)
    return dir.sample()

def log_propdist_rhos(prop,origin,diralpha=diralpha,tol=1e-5):
    dir = torch.distributions.Dirichlet(origin/tol)
    return dir.log_prob(prop)

# %%

def update(mus,sigs,rhos,lp,tol=1e-2,tol_rho =1e-5,burnin=False):
    mus_prop = prop_mus(mus,tol)
    lp_prop = lp_function(mus_prop,sigs,rhos)
    if torch.log(torch.rand(1))<(lp_prop-lp).item():
        mus = mus_prop
        lp = lp_prop

    sigs_prop = prop_sigs(sigs,tol)
    lp_prop = lp_function(mus,sigs_prop,rhos)
    if torch.log(torch.rand(1))<(lp_prop-lp).item():
        sigs = sigs_prop
        lp = lp_prop

    
    rhos_prop = prop_rhos(rhos,tol=tol_rho)
    #print(rhos,rhos_prop)
    lp_prop = lp_function(mus,sigs,rhos_prop)
    if torch.log(torch.rand(1))< (lp_prop - lp 
                                  + log_propdist_rhos(rhos,rhos_prop,tol=tol_rho) 
                                  - log_propdist_rhos(rhos_prop,rhos,tol=tol_rho)
                                  ).item():
        rhos = rhos_prop
        lp = lp_prop
    #print('r',torch.abs(rhos-rhos_prop).max())

    if burnin:
        order = torch.argsort(-rhos)
        mus,sigs,rhos = mus[order],sigs[order],rhos[order]
        lp= lp_function(mus,sigs,rhos)
            
    return mus,sigs,rhos,lp

# %%
thetas = []
lps = []

mus = torch.ones_like(diralpha)*(x.mean()/mu_a)
sigs = torch.ones_like(diralpha)
rhos = diralpha/diralpha.sum()

lp = lp_function(mus,sigs,rhos)
lp_new=lp

tols_burnin = linspace(.3,.01,5000)
for i in range(len(tols_burnin)):
    tol = tols_burnin[i]
    #while lp_new ==lp:
    mus,sigs,rhos,lp = update(mus,sigs,rhos,lp,tol=tol,tol_rho=tol*1e-3,burnin=True)

    thetas.append(torch.hstack((mus,sigs,rhos)))
    lps.append(lp)
        #print([a for a in (mus,sigs,rhos)],lp)
    if (i-1)%1000==0:
        print(lp.item())

burn=len(lps)

# %%
for i in range(20000):
    mus,sigs,rhos,lp = update(mus,sigs,rhos,lp,tol=1e-2,tol_rho=tol*1e-3)

    thetas.append(torch.hstack((mus,sigs,rhos)))
    lps.append(lp)

    if (i-1)%1000==0:
        print(i,[a[:5] for a in (mus,sigs,rhos)])
        print(lp.item())

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-3,ceil(x.max().item()/mu_a),2001).to(device)

p_gt = torch.exp(logprob_exp(xb,torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')

logprob_mixlognormal_hor =lambda x,mus,sigs,rhos: logprob_mixlognormal(x.reshape(-1,1),mus,sigs,rhos).reshape(-1)

i = torch.argmax(torch.stack(lps))
th_map = thetas[i]
p_map = torch.exp(logprob_mixlognormal_hor(xb,th_map[:Ncomp],th_map[Ncomp:2*Ncomp],th_map[2*Ncomp:]))
ax[0].plot(xb.cpu(),p_map.cpu(),label='MAP  KL={:.2E}'.format(KL(p_gt,p_map,xb)))

p = torch.exp(torch.vstack([logprob_mixlognormal_hor(xb,th[:Ncomp],th[Ncomp:2*Ncomp],th[2*Ncomp:]) for th in thetas[burn:]])).mean(axis=0)
ax[0].plot(xb.cpu(),p.cpu(),label='Bayesian reconstruction KL={:.2E}'.format(KL(p_gt,p,xb)))


ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/Bayes_NP_prod_N_{}_{}_E_{}_datapoints{}.png'.format(mu_a,sig_a,scale,N),dpi=600)

# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=(['mu_{}'.format(i) for i in range(Ncomp)]+['sigma_{}'.format(i) for i in range(Ncomp)]+['weight_{}'.format(i) for i in range(Ncomp)]))
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/prod_bayes_NP_N_{}_{}_E_{}_datapoints{}.csv'.format(mu_a,sig_a,scale,N))


# %%
with open('report_prod.csv', 'a') as file:
    file.write('N_{}_{}_E_{}.csv'.format(mu_a,sig_a,scale)+',npbayes,'+str(N)+','+str(KL(p_gt,p,xb).item())+','+str(KL(p_gt,p_map,xb).item())+'\n')

# %%



