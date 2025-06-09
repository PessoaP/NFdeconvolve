# %%

import torch
from numpy import loadtxt,sqrt,ceil,linspace
from matplotlib import pyplot as plt
from basis import *
import pandas as pd
import npbayes_class
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
N, shape = int(sys.argv[1]),int(sys.argv[2])
print(N,shape)

# %%
torch.manual_seed(42)
mu_a,sig_a = 10,1
shape,scale = shape,1
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float().to(device)

# %%
Ncomp=20
model = npbayes_class.Deconvolver(x,mu_a,sig_a,Ncomp=Ncomp)

# %%
start_chain = time.time()
thetas,lps = model.MCMC_chain()
end_chain = time.time()

df = pd.DataFrame((thetas).cpu().numpy(),columns=(['mu_{}'.format(i) for i in range(Ncomp)]+['sigma_{}'.format(i) for i in range(Ncomp)]+['weight_{}'.format(i) for i in range(Ncomp)]))
df['log_post'] = (lps).cpu().numpy()
df.to_csv('models/sum_NPbayes_'+filename.split('.')[0]+'datapoints_{}.csv'.format(N))


# %%
import csv
import os
csv_file = 'time_report.csv'

write_header = not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0

with open(csv_file, mode='a', newline='') as file:
    writer = csv.writer(file)

    # Write the header if needed
    if write_header:
        writer.writerow(['type', 'file', 'datapoints','time(s)'])

    # Write the new row (wrap in a list to make it a row)
    writer.writerow(['sum_mixture', filename.split('.')[0],N, end_chain - start_chain])