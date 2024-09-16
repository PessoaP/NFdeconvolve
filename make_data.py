import numpy as np

N=1000000
np.random.seed(42)

 
mu_a,sig_a = 10,1
shape,scale = .33,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

for shape in range(3,10):
    x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
    np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

##for snr in range(3,10):
    ##shape,scale=10/snr,1
    ##x = np.random.exponential(1,size=N)*np.random.gamma(shape,scale,size=N)
    ##np.savetxt('datasets/E_1_G_{}_{}.csv'.format(shape,scale),x)
for shape in range(3,10):
    x = np.random.normal(mu_a,sig_a,size=N)*np.random.gamma(shape,scale,size=N)
    np.savetxt('datasets/prod_N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)
    