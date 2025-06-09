# %%
import torch
from numpy import loadtxt,sqrt,ceil,floor
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
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)
x = torch.tensor(loadtxt('datasets/'+filename)[:N]).float()

# %%
a_distribution= torch.distributions.Normal(torch.tensor(mu_a),torch.tensor(sig_a))
model=nf_class.Deconvolver(x,a_distribution)

# %%
start_train = time.time()
model.train()
end_train = time.time()

# %%
torch.save(model.state_dict(), 'models/sum_nf_'+filename.split('.')[0]+'datapoints_{}.pt'.format(N))
del model

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
    writer.writerow(['sum_nf', filename.split('.')[0],N, end_train - start_train])