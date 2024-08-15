# %%
import torch
from numpy import loadtxt,sqrt,ceil
from matplotlib import pyplot as plt
from basis import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

import normflows as nf
import Zeta.nf_class as nf_class
from tqdm import tqdm

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
model = nf_class.NormalizingFlow_sim(device)

# %%
b_tensor_hor = torch.linspace(1e-3,ceil(x.max().item()/mu_a),10000).reshape(-1).to(device)
b_tensor_ver = (1.0*b_tensor_hor).reshape(-1,1)
db = (b_tensor_hor[1]-b_tensor_hor[0])

def log_likelihood(data,model,mu=torch.tensor(mu_a).to(device),sig=torch.tensor(sig_a).to(device)):
    lpb = model.log_prob(b_tensor_ver).reshape(-1,1)
    lpa = logprob_gaussian(data/b_tensor_ver,mu,sig)
    lp = lpb+lpa-torch.log(torch.abs(b_tensor_ver))

    return torch.log(db)+torch.logsumexp(lp,axis=0)

# %%
optimizer = torch.optim.Adam(model.parameters(), lr=.1/N)

loss_hist =[]
loss = -log_likelihood(x,model).mean()
print(loss)

# %%
max_iter =1000
show_iter = 100
for it in tqdm(range(max_iter)):
    optimizer.zero_grad()
    
    # Compute loss
    loss = -log_likelihood(x,model).mean()
    
    # Do backprop and optimizer step
    if ~(torch.isnan(loss) | torch.isinf(loss)):
        loss.backward()
        optimizer.step()
    
    # Log loss
    loss_hist.append(loss.item())
    
    # Plot learned distribution
    if (it + 1) % show_iter == 0:
        print(loss)
        xb = torch.linspace(1e-3,ceil(x.max().item()/mu_a),251).to(device)
        p_nf = torch.exp(model.log_prob(xb.reshape(-1,1))).detach()
        p_gt = torch.exp(logprob_exp(xb,torch.tensor(scale).to(device)))
        plt.plot(xb.cpu(),p_nf.cpu())
        #plt.scatter(xb.cpu(),p_nf.cpu())
        plt.plot(xb.cpu(),p_gt.cpu(),color='k')
        print(KL(p_gt,p_nf,xb))
        #plt.show()
    del loss

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-3,ceil(x.max().item()/mu_a),2001).to(device)

p_gt = torch.exp(logprob_exp(xb,torch.tensor(scale)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')
        
p_nf = torch.exp(model.log_prob(xb.reshape(-1,1))).detach()
ax[0].plot(xb.cpu(),p_nf.cpu(),label='NF  KL={:.2E}'.format(KL(p_gt,p_nf,xb)))
#ax[0].scatter(xb.cpu(),p_nf.cpu())

ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/nf_N_{}_{}_E_{}_datapoints{}.png'.format(mu_a,sig_a,scale,N),dpi=600)

# %%
torch.save(model.state_dict(), 'models/prod_nf_N_{}_{}_E_{}_datapoints{}.pt'.format(mu_a,sig_a,scale,N))

# %%
with open('report_prod.csv', 'a') as file:
    file.write('N_{}_{}_E_{}.csv'.format(mu_a,sig_a,scale)+',NF,'+str(N)+','+str(KL(p_gt,p_nf,xb).item())+','+str(KL(p_gt,p_nf,xb).item())+'\n')


