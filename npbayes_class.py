import torch
import normflows as nf  
from tqdm import tqdm
from basis import log_distribution,logprob_gaussian,normalize #make it part of thhis file for release
import warnings
from numpy import linspace,sqrt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def prop_mus(mus,tol=1e-2):
    return mus+tol*torch.randn_like(mus).to(device)

def prop_sigs(sigs,tol=1e-2):
    return torch.exp(torch.log(sigs)+tol*torch.randn_like(sigs).to(device))

def prop_rhos(rhos,tol=1e-5):
    dir = torch.distributions.Dirichlet(rhos/tol)
    return dir.sample()


def log_propdist_rhos(prop,origin,tol=1e-5):
    dir = torch.distributions.Dirichlet(origin/tol)
    return dir.log_prob(prop)

params2ten = lambda mus,sigs,rhos: (mus,torch.log(sigs),torch.log(rhos))

class Deconvolver:
    def __init__(self,data = None,
                 mu_a=0,sig_a=1,Ncomp=20,decay=.9,mult_factor=10,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),intervals = 20000):
        self.data = data.reshape(-1).to(device)
        self.mu_a=torch.tensor(mu_a).to(device)
        self.sig_a=torch.tensor(sig_a).to(device)
        self.N = data.size(0)
        self.device = device

        diralpha = ((decay)**torch.arange(Ncomp)).to(device)
        self.diralpha = normalize(diralpha)*mult_factor#/10
        self.dir_prior = torch.distributions.Dirichlet(self.diralpha)
        self.base_tensor = torch.ones_like(diralpha)

    def initial(self):
        mus  = self.base_tensor*(self.data.mean()-self.mu_a)
        sigs = self.base_tensor*self.data.std()
        rhos = normalize(self.diralpha)
        return mus, sigs, rhos

    def logprior(self,mus,sigs,rhos): 
        return (logprob_gaussian(mus,0*self.base_tensor,50*self.base_tensor).sum() 
                + logprob_gaussian(torch.log(sigs),self.base_tensor,self.base_tensor).sum() 
                - torch.log(sigs).sum()
                + self.dir_prior.log_prob(rhos))

    def logprob_mixgaussian(self,mus,sigs,rhos):
        return torch.logsumexp(logprob_gaussian(self.data,mus.reshape(-1,1),sigs.reshape(-1,1))
                               + torch.log(rhos.reshape(-1,1)),axis=0)

    def log_likelihood(self,mus,sigs,rhos):
        return self.logprob_mixgaussian(self.mu_a+mus,
                                        self.sig_a*torch.sqrt(torch.pow(sigs/self.sig_a,2)+1),
                                        rhos)

    def lp_function(self,mus,sigs,rhos):
        return self.log_likelihood(mus,sigs,rhos).sum() + self.logprior(mus,sigs,rhos)

    def update(self,mus,sigs,rhos,lp,tol=1e-2,tol_rho =1e-5,burnin=False):
        mus_prop = prop_mus(mus,tol)
        lp_prop = self.lp_function(mus_prop,sigs,rhos)
        if torch.log(torch.rand(1))<(lp_prop-lp).item():
            mus = mus_prop
            lp = lp_prop

        sigs_prop = prop_sigs(sigs,tol)
        lp_prop = self.lp_function(mus,sigs_prop,rhos)
        if torch.log(torch.rand(1))<(lp_prop-lp).item():
            sigs = sigs_prop
            lp = lp_prop

        
        rhos_prop = prop_rhos(rhos,tol=tol_rho)
        lp_prop = self.lp_function(mus,sigs,rhos_prop)
        if torch.log(torch.rand(1))< (lp_prop - lp 
                                    + log_propdist_rhos(rhos,rhos_prop,tol=tol_rho) 
                                    - log_propdist_rhos(rhos_prop,rhos,tol=tol_rho)
                                    ).item():
            rhos = rhos_prop
            lp = lp_prop
        #print('r',torch.abs(rhos-rhos_prop).max())

        if burnin:
            order = torch.argsort(-rhos)
            mus,sigs,rhos = mus[order],sigs[order],rhos[order]
            lp= self.lp_function(mus,sigs,rhos)
                
        return mus,sigs,rhos,lp
    
    def MCMC_chain(self,tols=.01,samples=10000,burnin_samples=5000,burnin_tol=.3,print_iter=False):
        thetas = []
        lps = []
        if self.N >500:
            mus,sigs,rhos = [a.detach() for a in self.MAP_estimate(print_iter=print_iter)[0]]
        else:
            mus,sigs,rhos = self.initial()
        rhos = .99*rhos +.01*normalize(torch.ones_like(rhos))

        lp = self.lp_function(mus,sigs,rhos)

        tols_burnin = linspace(burnin_tol,tols,burnin_samples)

        for i in tqdm(range(burnin_samples+samples)):
            if i<burnin_samples:
                tol = tols_burnin[i]
                mus,sigs,rhos,lp = self.update(mus,sigs,rhos,lp,tol=tol,tol_rho=tol*1e-3,burnin=True)
            else:
                mus,sigs,rhos,lp = self.update(mus,sigs,rhos,lp,tol=tol,tol_rho=tol*1e-3)

            thetas.append(torch.hstack((mus,sigs,rhos)))
            lps.append(lp)
            if ((i-1)%1000==0) and print_iter:
                print(i,[a[:5] for a in (mus,sigs,rhos)])
                print(lp.item())

        return torch.vstack(thetas), torch.stack(lps)
    
    def MAP_estimate(self,lr=1.,max_iter =5000,show_iter = 1000,scheduler_step=100,print_iter=False):
        Ncomp = self.diralpha.size(0)
        ten2params = lambda ten: (ten[:Ncomp], torch.exp(ten[Ncomp:2*Ncomp]), normalize(torch.exp(ten[2*Ncomp:])))

        tensor = torch.hstack((params2ten(*self.initial()))).to(self.device).requires_grad_(True)
        loss_function = lambda tensor: -self.lp_function(*ten2params(tensor))

        optimizer = torch.optim.Adam([tensor],lr=lr/sqrt(self.N))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=0.9)
        loss_hist =[]

        for it in tqdm(range(max_iter)):
            optimizer.zero_grad()
            loss = loss_function(tensor)

            if ~(torch.isnan(loss) | torch.isinf(loss)):
                loss.backward()
                optimizer.step()
                scheduler.step()

            loss_hist.append(loss.item())
            if ((it + 1) % show_iter == 0) and print_iter:
                print('Loss',-loss_hist[-1])
                print(it,[a[:5].cpu() for a in ten2params(tensor)])
        #self.trained=True

        return ten2params(tensor),-loss.item()