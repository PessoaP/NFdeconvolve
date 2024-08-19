# %%
import torch
from numpy import loadtxt,sqrt,ceil,floor
from matplotlib import pyplot as plt
from basis import *
import nf_class

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, shape = int(sys.argv[1]),int(sys.argv[2])
print(N,shape)

# %%
torch.manual_seed(42)
mu_a,sig_a = 10,1
shape,scale = shape,1
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float()#.to(device)

# %%
a_distribution= torch.distributions.Normal(torch.tensor(mu_a),torch.tensor(sig_a))
model=nf_class.Deconvolver(x,a_distribution)

# %%
model.train()

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = torch.linspace(1e-3,ceil(x.max().item()-mu_a/2),10000).reshape(-1)

p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape),torch.tensor(scale)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')
        
p_nf = model.get_pdf(xb)[1].cpu()
ax[0].plot(xb.cpu(),p_nf.cpu(),label='NF  KL={:.2E}'.format(KL(p_gt,p_nf,xb)))

ax[0].legend()
ax[0].set_ylabel('density')
ax[0].set_xlabel(r'$b$')

ax[1].hist(x.cpu(),bins=int(ceil(sqrt(x.size(0)))),alpha=.2,density=True)
ax[1].set_title('N={}'.format(N))
ax[1].set_xlabel(r'$x$')
plt.savefig('graphs/nf_N_{}_{}_G_{}_{}_datapoints{}.png'.format(mu_a,sig_a,shape,scale,N),dpi=600)

# %%
torch.save(model.state_dict(), 'models/sum_nf_'+filename.split('.')[0]+'datapoints_{}.pt'.format(N))

# %%
with open('report.csv', 'a') as file:
    file.write(filename+',NF,'+str(N)+','+str(KL(p_gt,p_nf,xb).item())+','+str(KL(p_gt,p_nf,xb).item())+'\n')

# %%



