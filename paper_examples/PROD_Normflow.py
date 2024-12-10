# %%
import torch
from numpy import loadtxt,sqrt,ceil,linspace
from matplotlib import pyplot as plt
import nf_class

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

a_distribution= torch.distributions.Normal(torch.tensor(mu_a).float().to(device),torch.tensor(sig_a).float().to(device))
model = nf_class.ProdDeconvolver(x,a_distribution,device)

# %%
model.train()

# %%
torch.save(model.state_dict(), 'models/prod_nf_'+filename.split('.')[0]+'datapoints_{}.pt'.format(N))

del model