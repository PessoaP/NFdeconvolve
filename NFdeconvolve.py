import torch
import normflows as nf  
from tqdm import tqdm
import warnings
from math import sqrt
l2 = lambda x: torch.sqrt((x*x).sum())
torch.manual_seed(0)

def create_nfm(device, K = 4, hidden_units = 32, hidden_layers_list = (1,2), tail_bound=30):
    latent_size = 1

    q0 = nf.distributions.DiagGaussian(1, trainable=False)
    flows = []
    for i in range(K):
        flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,tail_bound=tail_bound) for hidden_layers in hidden_layers_list]
        flows += [nf.flows.LULinearPermute(latent_size)]

    nfm = nf.NormalizingFlow(q0=q0, flows=flows)

    enable_cuda = True
    return nfm.to(device)

class NormalizingFlow_shifted:
    def __init__(self,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),center=0,width=1,tail_bound=30):
        self.nfm = create_nfm(device,tail_bound=tail_bound)
        self.center = torch.tensor(center).to(device)
        self.width = torch.tensor(width).to(device)
        self.zb = lambda x: (x-self.center)/self.width
        self.parameters = lambda: self.nfm.parameters()
        self.device = device
        
    def log_prob(self,x):
        return self.nfm.log_prob(self.zb(x))-torch.log(self.width)
    
    def sample(self,*args):
        zsamples = self.nfm.sample(*args)
        return self.center+self.width*zsamples[0], zsamples[1]-torch.log(self.width)
    
    def state_dict(self):
        g = self.nfm.state_dict().copy()
        g['center,width'] = torch.stack((self.center,self.width))
        return g
    
    def load_state_dict(self,dict):
        self.center,self.width = dict.pop('center,width').to(self.device)
        self.zb = lambda x: (x-self.center)/self.width
        return self.nfm.load_state_dict(dict)


class Deconvolver:
    def __init__(self,data = None,
                 a_distribution = None,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                 intervals = 20000,
                 tail_bound = 30):
        self.a_distribution = a_distribution#.to(device)
        
        if (data is None) or (a_distribution is None):
            self.nfs = NormalizingFlow_shifted(device,torch.zeros(1,device=device),torch.ones(1,device=device),tail_bound=tail_bound)

        else:
            mu_a,sig_a = self.a_distribution.mean,self.a_distribution.stddev
            self.data = data.reshape(-1).to(device)
            self.N = data.size(0)
            self.device = device

            self.b_vals = torch.linspace(data.min().item()-mu_a-3*sig_a, data.max().item()-mu_a+3*sig_a,intervals).to(device).reshape(-1,1)
            self.log_db = torch.log(self.b_vals[1]-self.b_vals[0])

            width = self.data.std().item() 
            width *= max(1/4, torch.sqrt(1-torch.pow(sig_a/self.data.std(),2)).item() )
            self.nfs = NormalizingFlow_shifted(device,
                                               (self.data.mean()-mu_a).item(),
                                               width,tail_bound=tail_bound)

        self.device = device
        self.trained = False

    def log_likelihood(self):
        if self.a_distribution is None:
            return warnings.warn('The distribution was not provided. We cannot calculate likelihoods')
        lpb = self.nfs.log_prob(self.b_vals).reshape(-1,1)
        lpa = self.a_distribution.log_prob(self.data-self.b_vals)
        return self.log_db + torch.logsumexp(lpb+lpa,axis=0)

    
    def train(self,grad_tol = .05, max_iter = 5000,show_iter = 100,print_iter=False):
        gradient_norm = 2*grad_tol
        if self.trained:
            warnings.warn("The network has already been trained. Proceeding will overwrite the previous training.")
        if self.data.numel == 0:
            warnings.warn("No observation data provided. Training cannot proceed.")
            return
        if not torch.cuda.is_available():
            warnings.warn(
                "CUDA-compatible GPU not detected. NFdeconvolve is optimized for GPU training, and performance may be significantly slower on a CPU."
            )

        optimizer = torch.optim.Adam(self.nfs.parameters(), lr=.1/self.N)
        loss_hist =[]

        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            loss = -(self.log_likelihood().mean())
            
            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()

            loss_hist.append(loss.item())
            if ((it + 1) % show_iter == 0):
                gradient_norm = l2(torch.stack([l2(p.grad) for p in self.nfs.parameters()])).item()
                if print_iter:
                    print('Loss:',loss_hist[-1],'   gradient:',gradient_norm)
                if gradient_norm < grad_tol:
                    print(f'Stopping early at iteration {it + 1} due to small gradient norm: {gradient_norm}')
                    break

        self.trained=True

    def get_pdf(self, values = None,log_prob=False):
        if values is None:
            values = self.b_vals*1.0
        lp = self.nfs.log_prob(values.reshape(-1,1).to(self.device)).detach()
        if log_prob:
            return values.reshape(-1),lp
        return values.reshape(-1), torch.exp(lp)

    def state_dict(self):
        g = self.nfs.state_dict().copy()
        g['b_array:min,max,intervals'] = torch.tensor((self.b_vals[0].float(),
                                                       self.b_vals[-1].float(),
                                                       self.b_vals.numel()))
        return g
    
    def load_state_dict(self,dict):
        xb_min,xb_max,intervals = dict.pop('b_array:min,max,intervals')#.to(self.device)
        self.b_vals = torch.linspace(xb_min,xb_max,intervals.int().item()).to(self.device).reshape(-1,1)
        self.log_db = torch.log(self.b_vals[1]-self.b_vals[0])

        self.nfs.load_state_dict(dict)
        self.trained=True


class log_distribution:
    """
    A wrapper class for transforming a probability distribution into its equivalent in log space.

    Attributes:
        dist (torch.distributions.Distribution): The base probability distribution.
        mean (float): Approximate mean of the log-sampled values. Computed during initialization.
        stddev (float): Approximate standard deviation of the log-sampled values. Computed during initialization.

    Methods:
        sample(args):
            Samples values from the base distribution in log space.
        log_prob(lx):
            Computes the log-probability of a value in log space.
    """
    def __init__(self, distribution):
        self.dist = distribution

        lx = self.sample((10000,))
        self.mean = lx.mean()
        self.stddev = lx.std()
        
    def sample(self,*args):
        return torch.log(self.dist.sample(*args))

    def log_prob(self,lx):
        return self.dist.log_prob(torch.exp(lx)) + lx
    
class ProdDeconvolver:
    def __init__(self,data = None,a_distribution = None,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),intervals = 20000):
        if (data is None) or (a_distribution is None):
            self.subdeconvolver = Deconvolver(data,a_distribution,device)
        else:
            self.la_distribution = log_distribution(a_distribution)
            self.ldata = torch.log(data)
            self.subdeconvolver = Deconvolver(self.ldata,self.la_distribution,device,intervals)
            self.device = device
        self.lb_vals = torch.linspace(min((1e-3,torch.exp(self.subdeconvolver.b_vals[0]).item())),
                                           torch.exp(self.subdeconvolver.b_vals[-1]).item(),
                                           intervals).to(self.device).reshape(-1,1)
        
    def train(self,grad_tol=.05,max_iter =1000,show_iter = 100):
        self.subdeconvolver.train(grad_tol,max_iter,show_iter)

    def state_dict(self):
        return self.subdeconvolver.state_dict()
    
    def load_state_dict(self,dict):
        self.subdeconvolver.load_state_dict(dict)
        self.lb_vals = torch.linspace(min((1e-3,torch.exp(self.subdeconvolver.b_vals[0]).item())),
                                           torch.exp(self.subdeconvolver.b_vals[-1]).item(),
                                           self.subdeconvolver.b_vals.numel()).to(self.device).reshape(-1,1)

    def get_pdf(self, values = None,log_prob=False):
        if values is None:
            values = self.lb_vals
        lvalues = torch.log(values).to(self.device)
        lp = self.subdeconvolver.get_pdf(lvalues,log_prob=True)[1] - lvalues.reshape(-1)
        if log_prob:
            return values.reshape(-1),lp
        return values.reshape(-1), torch.exp(lp)