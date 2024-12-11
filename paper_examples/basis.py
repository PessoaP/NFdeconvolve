import torch    
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from numpy import percentile
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def logprob_mixgaussian(x,mus,sigs,rhos):
    return torch.logsumexp(logprob_gaussian(x.reshape(-1),mus.reshape(-1,1),sigs.reshape(-1,1))
                           + torch.log(rhos.reshape(-1,1)),axis=0)


def comparative_plot(name,NF,th,pb_gt,map_index,x,b,tails=None,loc=2):
    if tails is not None :
        xb_nf,pb_nf = NF.get_pdf(torch.linspace(tails[0],tails[1],251,device=NF.device))
    else:
        xb_nf,pb_nf = NF.get_pdf()

    pb_npmap = torch.exp(logprob_mixgaussian(xb_nf,th[map_index][:20],th[map_index][20:40],th[map_index][40:60]))
    pb_rec = torch.stack([torch.exp(logprob_mixgaussian(xb_nf,thi[:20],thi[20:40],thi[40:60])) for thi in th[5000:]]).mean(axis=0)
    p = pb_gt(xb_nf)

    fig,ax = plt.subplots(2,1,figsize=(6,7))

    h = ax[0].hist(x, alpha=.7,density=True,label='Data',bins=49)
    ax[0].hist(b, alpha=.4,density=True,label='Signal',bins=h[1])

    #plt.plot(xb_nf.detach().cpu(),pb_npmap.detach().cpu())
    ax[1].plot(xb_nf.detach().cpu(),pb_rec.detach().cpu(),label='Gaussian Mixture',linewidth=2)
    ax[1].plot(xb_nf.detach().cpu(),pb_nf.detach().cpu(),label='NFdeconvolve',linewidth=2,color='r')

    ax[1].plot(xb_nf.detach().cpu(),p.detach().cpu(),color='k',label = 'Ground truth',linewidth=1)
 
    ax[1].set_ylabel('Density',fontsize=12)
    if tails is not None :
        [axi.set_xlim(*tails) for axi in ax]

    [axi.legend(fontsize=10,loc=loc) for axi in ax]
    plt.tight_layout()
    plt.savefig(name+'_example.png',dpi=500)

    print(name, 'models')
    print('NFdeconvolve',KL(p.cpu(),pb_nf.cpu(),xb_nf.cpu()))
    print('Reconstruct  ',KL(p.cpu(),pb_rec.cpu(),xb_nf.cpu()))
    print('     MAP     ',KL(p.cpu(),pb_npmap.cpu(),xb_nf.cpu()))

class log_distribution:
    def __init__(self, distribution):
        self.dist = distribution

        lx = self.sample((10000,))
        self.mean = lx.mean()
        self.stddev = lx.std()
        
    def sample(self,*args):
        return torch.log(self.dist.sample(*args))

    def log_prob(self,lx):
        return self.dist.log_prob(torch.exp(lx)) + lx