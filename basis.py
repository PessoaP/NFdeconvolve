import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

lsqrt2pi = (torch.log(torch.tensor(2*torch.pi))/2).to(device)

def logprob_gaussian(x,mu,sig):
    return -(((x-mu)/sig)**2)/2 - torch.log(sig) - lsqrt2pi

def logprob_gamma(x,k,th):
    return ((k-1)*torch.log(x) - x/th - k*torch.log(th) - torch.lgamma(k))

def logprob_dirichlet(rho,alpha):
    return torch.lgamma(alpha.sum()) - torch.lgamma(alpha).sum()+ ((alpha-1)*torch.log(rho)).sum()

logprob_exp = lambda x,scale: -x/scale-torch.log(scale)

def KL(p,q,x):
    dx = x[1]-x[0]
    p,q = p[p>0],q[p>0]
    return dx*(p*(torch.log(p)-torch.log(q))).sum()

normalize = lambda x:x/x.sum()
