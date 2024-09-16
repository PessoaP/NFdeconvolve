import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

lsqrt2pi = (torch.log(torch.tensor(2*torch.pi))/2).to(device)

def logprob_gaussian(x,mu,sig):
    return -(((x-mu)/sig)**2)/2 - torch.log(sig) - lsqrt2pi

def logprob_gamma(x,k,th):
    return ((k-1)*torch.log(x) - x/th - k*torch.log(th) - torch.lgamma(k))

def logprob_dirichlet(rho,alpha):
    return torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()+ ((alpha-1)*torch.log(rho)).sum()

logprob_exp = lambda x,scale: -x/scale-torch.log(scale)
logprob_lexp = lambda lx,scale: logprob_exp(torch.exp(lx),scale) +lx
logprob_lgamma = lambda lx,k,th: logprob_gamma(torch.exp(lx),k,th) +lx


def KL(p,q,x):
    dx = x[1]-x[0]
    p,q = p[p>0],q[p>0]
    return dx*(p*(torch.log(p)-torch.log(q))).sum()

normalize = lambda x:x/x.sum()

class log_distribution:
    def __init__(self, distribution):
        self.dist = distribution

        ##This is a rough estimation, it just made to help training, where it does not need to be 
        lx = self.sample((10000,))
        self.mean = lx.mean()
        self.stddev = lx.std()
        
    def sample(self,args):
        return torch.log(self.dist.sample(args))

    def log_prob(self,lx):
        return self.dist.log_prob(torch.exp(lx)) + lx