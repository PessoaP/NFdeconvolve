# %%
import pandas as pd
import nf_class
import torch
from matplotlib import pyplot as plt

from tqdm import tqdm
torch.manual_seed(0)

# %%
class nfm_to_torch_distributions():
    def __init__(self,data):
        self.nfm = nf_class.NormalizingFlow_shifted(center = data.mean(),width = data.std())
        print('Training shifted')
        optimizer = torch.optim.Adam(self.nfm.parameters(), lr=1/data.numel())
        loss_hist =[]

        l2 = lambda x: torch.sqrt((x*x).sum())
        for it in tqdm(range(5000)):
            optimizer.zero_grad()
            loss = -(self.nfm.log_prob(data.reshape(-1,1)).mean())
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            loss_hist.append(loss.item())
            if ((it + 1) % 100 == 0):
                gradient_norm = l2(torch.stack([l2(p.grad) for p in self.nfm.parameters()])).item()
                print('Loss:',loss_hist[-1],'   gradient:',gradient_norm)
                if gradient_norm < .05:
                    print(f'Stopping early at iteration {it + 1} due to small gradient norm: {gradient_norm}')
                    break
            del loss 
            
        for param in self.nfm.parameters():
            param.requires_grad = False
            
        g = self.nfm.sample(10000)[0]
        self.mean,self.stddev = g.mean(),g.std()

    def sample(self,*args):
        self.nfm.sample(*args)

    def log_prob(self,x):
        fshape = x.shape
        return self.nfm.log_prob(x.reshape(-1,1)).reshape(fshape)


# %%
control = pd.read_csv('FCdata/complete_d=0.33.csv')['FL1-A'].to_numpy()
control = torch.tensor(control).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()

# %%
a_dist = nfm_to_torch_distributions(control)
torch.save(a_dist.nfm.state_dict(),'models/FCdata_control_NF.pt')

# %%
lowstress = pd.read_csv('FCdata/complete_d=0.23.csv')['FL1-A'].to_numpy()
lowstress = torch.tensor(lowstress).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()

lowstress_deconvolver = nf_class.Deconvolver(lowstress[:10000],a_dist)
lowstress_deconvolver.train()
torch.save(lowstress_deconvolver.state_dict(),'models/FCdata_lowstress_NF.pt')

del lowstress_deconvolver

# %%
highstress = pd.read_csv('FCdata/complete_d=0.12.csv')['FL1-A'].to_numpy()
highstress = torch.tensor(highstress).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')).float()

highstress_deconvolver = nf_class.Deconvolver(highstress[:10000],a_dist)
highstress_deconvolver.train()
torch.save(highstress_deconvolver.state_dict(),'models/FCdata_highstress_NF.pt')


