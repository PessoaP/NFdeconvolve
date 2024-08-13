# %%
import torch
from numpy import loadtxt,sqrt,ceil
from matplotlib import pyplot as plt
from basis import *

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
import normflows as nf
from tqdm import tqdm

# Define flows
K = 4
torch.manual_seed(0)

latent_size = 1
hidden_units = 32
hidden_layers = 1

flows = []
for i in range(K):
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,tail_bound=30)]
    flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, 2*hidden_layers, hidden_units,tail_bound=30)]
    flows += [nf.flows.LULinearPermute(latent_size)]

# Set base distribuiton
q0 = nf.distributions.DiagGaussian(1, trainable=False)
    
# Construct flow model
nfm = nf.NormalizingFlow(q0=q0, flows=flows)

# Move model on GPU if available
enable_cuda = True
model = nfm.to(device)

class NormalizingFlow_shifted:
    def __init__(self,nfm,mu=0,sig=1):
        self.nfm = nfm
        self.mu = torch.tensor(mu).to(device)
        self.sig = torch.tensor(sig).to(device)
        self.zb = lambda x: (x-self.mu)/self.sig
        self.parameters = lambda: nfm.parameters()
        
    def log_prob(self,x):
        return nfm.log_prob(self.zb(x))-torch.log(self.sig)
    
    def sample(self,*args):
        logsamples = nfm.sample(*args)
        return self.mu+self.sig*logsamples[0], logsamples[1]-torch.log(self.sig)
    
    def state_dict(self):
        g = nfm.state_dict().copy()
        g['mean,scale'] = torch.stack((self.mu,self.sig))
        return g
    
    def load_state_dict(self,dict):
        self.mu,self.sig = dict.pop('mean,scale').to(device)
        self.zb = lambda x: (x-self.mu)/self.sig
        return nfm.load_state_dict(dict)

model = NormalizingFlow_shifted(nfm.to(device),x.mean()-mu_a,x.std())



# %%
b_tensor_hor = torch.linspace(1e-3,ceil(x.max().item()-mu_a/2),10000).reshape(-1).to(device)
b_tensor_ver = (1.0*b_tensor_hor).reshape(-1,1)
db = (b_tensor_hor[1]-b_tensor_hor[0])
print(db)

def log_likelihood(data,model,mu=torch.tensor(mu_a).to(device),sig=torch.tensor(sig_a).to(device)):
    lpb = model.log_prob(b_tensor_ver).reshape(-1,1)
    lpa = logprob_gaussian(data-b_tensor_ver,mu,sig)

    return torch.log(db)+torch.log(torch.sum(torch.exp(lpb+lpa),axis=0))


# %%
optimizer = torch.optim.Adam(model.parameters(), lr=.1/N)

loss_hist =[]
loss = -log_likelihood(x[:100],model).mean()
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
        xb = 1.0*b_tensor_hor 
        p_nf = torch.exp(model.log_prob(xb.reshape(-1,1))).detach()
        p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape).to(device),torch.tensor(scale).to(device)))
        print('kl',KL(p_gt,p_nf,xb),'loss',loss_hist[-1])
        #plt.show()

# %%
fig,ax = plt.subplots(1,2,figsize=(2*6.4,4.8))
xb = 1.0*b_tensor_hor 

p_gt = torch.exp(logprob_gamma(xb,torch.tensor(shape).to(device),torch.tensor(scale).to(device)))
ax[0].plot(xb.cpu(),p_gt.cpu(),label='Ground Truth ', lw = 2.0,color='k')
        
p_nf = torch.exp(model.log_prob(xb.reshape(-1,1))).detach()
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



