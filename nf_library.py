import torch
import normflows as nf  


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

    # Set base distribuiton
    q0 = nf.distributions.DiagGaussian(1, trainable=False)
        
    # Construct flow model
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


class NormalizingFlow_sim:
    def __init__(self,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        self.nfm = create_nfm(device)
    def parameters(self):
        return self.nfm.parameters()
    def log_prob(self,x):
        return torch.logaddexp(self.nfm.log_prob(x),self.nfm.log_prob(-x))
    def sample(self,*args):
        logsamples = self.nfm.sample(*args)
        return torch.abs(logsamples[0]),(self.log_prob(logsamples[0])).reshape(-1)#2*logsamples[1]
    def state_dict(self):
        return self.nfm.state_dict().copy()

    
    def load_state_dict(self,dict):
        return self.nfm.load_state_dict(dict)
