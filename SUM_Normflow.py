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
torch.save(model.state_dict(), 'models/sum_nf_'+filename.split('.')[0]+'datapoints_{}.pt'.format(N))

