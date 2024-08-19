import torch
import normflows as nf  
from tqdm import tqdm
from basis import log_distribution


torch.manual_seed(0)

def create_nfm(device):
    K = 4
    latent_size = 1
    hidden_units = 32
    hidden_layers = 1

    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,tail_bound=30)]
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, 2*hidden_layers, hidden_units,tail_bound=30)]
        flows += [nf.flows.LULinearPermute(latent_size)]

    q0 = nf.distributions.DiagGaussian(1, trainable=False)
    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    # Move model on GPU if available
    enable_cuda = True
    return nfm.to(device)


class NormalizingFlow_shifted:
    def __init__(self,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),mu=0,sig=1):
        self.nfm = create_nfm(device)
        self.mu = torch.tensor(mu).to(device)
        self.sig = torch.tensor(sig).to(device)
        self.zb = lambda x: (x-self.mu)/self.sig
        self.parameters = lambda: self.nfm.parameters()
        self.device = device
        
    def log_prob(self,x):
        return self.nfm.log_prob(self.zb(x))-torch.log(self.sig)
    
    def sample(self,*args):
        logsamples = self.nfm.sample(*args)
        return self.mu+self.sig*logsamples[0], logsamples[1]-torch.log(self.sig)
    
    def state_dict(self):
        g = self.nfm.state_dict().copy()
        g['mean,scale'] = torch.stack((self.mu,self.sig))
        return g
    
    def load_state_dict(self,dict):
        self.mu,self.sig = dict.pop('mean,scale').to(self.device)
        self.zb = lambda x: (x-self.mu)/self.sig
        return self.nfm.load_state_dict(dict)


class Deconvolver:
    def __init__(self,data,a_distribution,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.a_distribution = a_distribution#.to(device)
        mu_a,sig_a = self.a_distribution.mean,self.a_distribution.stddev
        
        self.data = data.reshape(-1).to(device)
        self.N=data.size(0)
        self.device = device

        self.b_vals = torch.linspace(data.min().item()-mu_a-3*sig_a, data.max().item()-mu_a+3*sig_a,20000).to(device).reshape(-1,1)
        self.log_db = torch.log(self.b_vals[1]-self.b_vals[0])

        self.nfs = NormalizingFlow_shifted(device,self.data.mean()-mu_a,self.data.std())
        self.trained=False

    def log_likelihood(self):
        lpb = self.nfs.log_prob(self.b_vals).reshape(-1,1)
        lpa = self.a_distribution.log_prob(self.data-self.b_vals)
        return self.log_db + torch.logsumexp(lpb+lpa,axis=0)
    
    def train(self,max_iter =1000,show_iter = 100):
        optimizer = torch.optim.Adam(self.nfs.parameters(), lr=.1/self.N)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
        loss_hist =[]

        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            loss = -self.log_likelihood().mean()

            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                scheduler.step()

            loss_hist.append(loss.item())
            if (it + 1) % show_iter == 0:
                print('Loss',loss_hist[-1])
        self.trained=True

    def get_pdf(self, values = None,log_prob=False):
        if values is None:
            values = self.b_vals
        lp = self.nfs.log_prob(values.reshape(-1,1).to(self.device)).detach()
        if log_prob:
            return values.reshape(-1),lp
        return values.reshape(-1), torch.exp(lp)

    def state_dict(self):
        return self.nfs.state_dict()
    
    def load_state_dict(self,dict):
        self.trained=True
        self.nfs.load_state_dict(dict)
    

class ProdDeconvolver:
    def __init__(self,data,a_distribution,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.la_distribution = log_distribution(a_distribution)
        self.ldata = torch.log(data)
        self.subdeconvolver = Deconvolver(self.ldata,self.la_distribution,device)
        self.device = device
        self.lb_vals = torch.linspace(min((1e-3,torch.exp(self.subdeconvolver.b_vals[0]).item())),
                                           torch.exp(self.subdeconvolver.b_vals[-1]).item(),
                                           20000).to(device).reshape(-1,1)
        
    def train(self,max_iter =1000,show_iter = 100):
        self.subdeconvolver.train(max_iter,show_iter)

    def state_dict(self):
        return self.subdeconvolver.state_dict()
    
    def load_state_dict(self,dict):
        self.subdeconvolver.load_state_dict(dict)

    def get_pdf(self, values = None,log_prob=False):
        if values is None:
            values = self.lb_vals
        lvalues = torch.log(values).to(self.device)
        lp = self.subdeconvolver.get_pdf(lvalues,log_prob=True)[1] - lvalues
        if log_prob:
            return values.reshape(-1),lp
        return values.reshape(-1), torch.exp(lp)