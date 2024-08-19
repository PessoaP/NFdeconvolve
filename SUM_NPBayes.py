# %%
import torch
from numpy import loadtxt,sqrt,ceil,linspace
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
#N = 10000
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float().to(device)

# %%
def logprob_mixgaussian(x,mus,sigs,rhos):
    return torch.logsumexp(logprob_gaussian(x,mus.reshape(-1,1),sigs.reshape(-1,1))
                           +torch.log(rhos.reshape(-1,1)),axis=0)


def log_likelihood(data,
                   mus,sigs,rhos,
                   mu_a=torch.tensor(mu_a).to(device),sig_a=torch.tensor(sig_a).to(device)):
    return logprob_mixgaussian(data,mus+mu_a,torch.sqrt(sigs**2+sig_a**2),rhos)


# %%

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
def prop_mus(mus,tol=1e-2):
    return mus+tol*torch.randn(Ncomp).to(device)

def prop_sigs(sigs,tol=1e-2):
    return torch.exp(torch.log(sigs)+tol*torch.randn(Ncomp).to(device))

def prop_rhos(rhos,tol=1e-5):
    dir = torch.distributions.Dirichlet(rhos/tol)
    return dir.sample()

def log_propdist_rhos(prop,origin,tol=1e-5):
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

mus = torch.ones_like(diralpha)*(x.mean()-mu_a)
sigs = torch.ones_like(diralpha)*x.std()
rhos = diralpha/diralpha.sum()
#mus,sigs,rhos = make_initial(x)


lp = lp_function(mus,sigs,rhos)
lp_new=lp

tols_burnin = linspace(.3,.01,5000)
for i in range(len(tols_burnin)):
    tol = tols_burnin[i]
    mus,sigs,rhos,lp = update(mus,sigs,rhos,lp,tol=tol,tol_rho=tol*1e-3,burnin=True)

    thetas.append(torch.hstack((mus,sigs,rhos)))
    lps.append(lp)
    if (i-1)%1000==0:
        print('logpost=',lp.item(),'logprior=',logprior(mus,sigs,rhos))

burn=len(lps)


# %%

for i in range(10000):
    mus,sigs,rhos,lp = update(mus,sigs,rhos,lp,tol=tol,tol_rho=tol*1e-3)

    thetas.append(torch.hstack((mus,sigs,rhos)))
    lps.append(lp)

    if (i-1)%1000==0:
        print(i,[a[:5] for a in (mus,sigs,rhos)])
        print(lp.item())

# %%

fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb =  torch.linspace(1e-3,ceil(x.max().item()-mu_a/2),10000).reshape(-1).to(device)
print(xb[0]-xb[1])

p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape).to(device),torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')



i = torch.argmax(torch.stack(lps))
th_map = thetas[i]
p_map = torch.exp(logprob_mixgaussian(xb,th_map[:Ncomp],th_map[Ncomp:2*Ncomp],th_map[2*Ncomp:]))
ax[0].plot(xb.cpu(),p_map.cpu(),label='MAP  KL={:.2E}'.format(KL(p_gt,p_map,xb)))

#p = torch.exp(torch.vstack([logprob_mixgaussian(xb,th[:Ncomp],th[Ncomp:2*Ncomp],th[2*Ncomp:]) for th in thetas[burn:]])).mean(axis=0)
lp = torch.logsumexp(torch.vstack([logprob_mixgaussian(xb,th[:Ncomp],th[Ncomp:2*Ncomp],th[2*Ncomp:]) for th in thetas[burn:]]),axis=0)-torch.log(torch.tensor(len(thetas[burn:]))).to(device)
p = torch.exp(lp)
ax[0].plot(xb.cpu(),p.cpu(),label='Bayesian reconstruction KL={:.2E}'.format(KL(p_gt,p,xb)))


ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/Bayes_NP_N_{}_{}_G_{}_{}_datapoints{}.png'.format(mu_a,sig_a,shape,scale,N),dpi=600)

# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=(['mu_{}'.format(i) for i in range(Ncomp)]+['sigma_{}'.format(i) for i in range(Ncomp)]+['weight_{}'.format(i) for i in range(Ncomp)]))
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/sum_npbayes_'+filename.split('.')[0]+'datapoints_{}.csv'.format(N))

# %%
with open('report.csv', 'a') as file:
    file.write(filename+',npbayes,'+str(N)+','+str(KL(p_gt,p,xb).item())+','+str(KL(p_gt,p_map,xb).item())+'\n')
