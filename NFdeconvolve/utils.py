import torch
import normflows as nf  
torch.manual_seed(0)

class AffineSimple(nf.flows.Flow):
    """
    A simple affine transformation flow: z ↦ scale * z + shift

    Parameters:
        shift (float or tensor): Additive offset (default: 0).
        scale (float or tensor): Multiplicative scale (default: 1).
    """
    def __init__(self, shift=0.0, scale=1.0):
        super().__init__()
        self.register_buffer("shift", torch.tensor(shift))
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, z):
        z = self.scale * z + self.shift
        log_det = torch.log(self.scale).expand(z.shape[0])
        return z, log_det

    def inverse(self, z):
        z = (z - self.shift) / self.scale
        log_det = -torch.log(self.scale).expand(z.shape[0])
        return z, log_det
    

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