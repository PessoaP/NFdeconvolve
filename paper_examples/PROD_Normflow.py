# %%
import torch
from numpy import loadtxt,sqrt,ceil,linspace
from matplotlib import pyplot as plt
import nf_class
import time

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
start_train = time.time()
model.train()
end_train = time.time()

# %%
torch.save(model.state_dict(), 'models/prod_nf_'+filename.split('.')[0]+'datapoints_{}.pt'.format(N))
del model

# %%
import csv
import os
csv_file = 'time_report.csv'

write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write the new row (wrap in a list to make it a row)
    writer.writerow(['prod_nf', filename.split('.')[0].split('_', 1)[1],N,end_train - start_train])