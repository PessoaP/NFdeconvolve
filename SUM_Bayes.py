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
#N = 10000
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float().to(device)

# %%
b_tensor_hor = torch.linspace(1e-3,ceil(x.max().item()-mu_a/2),10000).reshape(-1).to(device)
b_tensor_ver = (1.0*b_tensor_hor).reshape(-1,1)
db = (b_tensor_hor[1]-b_tensor_hor[0])
print(db)

def log_likelihood(data,shape,scale,mu=torch.tensor(mu_a).to(device),sig=torch.tensor(sig_a).to(device),N_samples=10000):
    lpb = logprob_gamma(b_tensor_ver,shape,scale)
    lpa = logprob_gaussian(data-b_tensor_ver,mu,sig)
    return torch.log(db)+torch.log(torch.sum(torch.exp(lpb+lpa),axis=0))


def logprior(lshape,lscale):
    return logprob_gaussian(lshape,torch.tensor(0),torch.tensor(100))+logprob_gaussian(lscale,torch.tensor(0),torch.tensor(100))

# %%
th_gt = torch.tensor((shape,scale)).float().to(device)
log_likelihood(x,*th_gt).sum()+ logprior(*torch.log(th_gt))


# %%
sh,sp = torch.meshgrid(torch.linspace(1e-3,10,26),torch.linspace(1e-3,6,26))
sh,sp = sh.reshape(-1).to(device), sp.reshape(-1).to(device)
lps = [log_likelihood(x,shi,spi).sum() for shi,spi in zip(sh,sp)]
i = torch.argmax(torch.stack(lps))
th = torch.stack((sh[i],sp[i]))
lth = torch.log(th)
lp = lps[i] + logprior(*lth)

# %%
lp_function = lambda lth: log_likelihood(x,*torch.exp(lth)).sum() + logprior(*lth)

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
        print(thetas[-1],lp)

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = 1.0*b_tensor_hor 

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
plt.savefig('graphs/Bayes_N_{}_{}_G_{}_{}_datapoints{}.png'.format(mu_a,sig_a,shape,scale,N),dpi=600)

# %%
df = pd.DataFrame(torch.vstack(thetas).cpu().numpy(),columns=['shape','scale'])
df['log_post'] = torch.stack(lps).cpu().numpy()
df.to_csv('models/sum_bayes_'+filename.split('.')[0]+'datapoints_{}.csv'.format(N))

# %%
with open('report.csv', 'a') as file:
    file.write(filename+',bayes,'+str(N)+','+str(KL(p_gt,p,xb).item())+','+str(KL(p_gt,p_map,xb).item())+'\n')




