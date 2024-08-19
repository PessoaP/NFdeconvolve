# %%
import torch
from numpy import loadtxt,sqrt,ceil,linspace
from matplotlib import pyplot as plt
from basis import *
import nf_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, snr = int(sys.argv[1]),float(sys.argv[2])
print(N,1/snr)

shape,scale=10/snr,1


# %%
torch.manual_seed(42)
expo_scale = 1
x = torch.tensor(loadtxt('datasets/E_{}_G_{}_{}.csv'.format(expo_scale,shape,scale))[:N]).float().to(device)


# %%
#lx = torch.log(x)
#a_distribution= log_distribution(torch.distributions.Exponential(expo_scale*1.0))
#model = nf_class.ProdDeconvolver(x,a_distribution,device)


a_distribution= torch.distributions.Exponential(expo_scale*1.0)
model = nf_class.ProdDeconvolver(x,a_distribution,device)

# %%
model.train()


# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-3,ceil(x.max().item()),10000).to(device)

p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape).to(device),torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')
        
p_nf = model.get_pdf(xb)[1]
ax[0].plot(xb.cpu(),p_nf.cpu(),label='NF  KL={:.2E}'.format(KL(p_gt,p_nf,xb)))

ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/Prod_nf_N_{}_E_{}_G_{}_{}.png'.format(N,expo_scale,10/snr,1),dpi=600)
# %%
torch.save(model.state_dict(), 'models/Prod_nf_N_{}_E_{}_G_{}_{}'.format(N,expo_scale,10/snr,1)+'datapoints_{}.pt'.format(N))


# %%
with open('report_prod.csv', 'a') as file:
    file.write('E_{}_G_{}_{}.csv'.format(expo_scale,10/snr,1)+',NFs,'+str(N)+','+str(KL(p_gt,p_nf,xb).item())+','+str(KL(p_gt,p_nf,xb).item())+'\n')
