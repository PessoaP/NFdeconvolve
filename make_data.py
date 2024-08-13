import numpy as np

N=1000000
np.random.seed(42)

 
mu_a,sig_a = 10,1
shape,scale = .33,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 3,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 4,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 5,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 6,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 7,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 8,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)

shape,scale = 9,1 
x = np.random.normal(mu_a,sig_a,size=N)+np.random.gamma(shape,scale,size=N)
np.savetxt('datasets/N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale),x)




mu_a,sig_a = 100,10
x = np.random.normal(mu_a,sig_a,size=N)*np.random.exponential(size=N)
np.savetxt('datasets/N_{}_{}_E_1.csv'.format(mu_a,sig_a),x)

mu_a,sig_a = 100,(10*np.sqrt(10)).round(2)
x = np.random.normal(mu_a,sig_a,size=N)*np.random.exponential(size=N)
np.savetxt('datasets/N_{}_{}_E_1.csv'.format(mu_a,sig_a),x)

mu_a,sig_a = 100,(10/np.sqrt(10)).round(2)
x = np.random.normal(mu_a,sig_a,size=N)*np.random.exponential(size=N)
np.savetxt('datasets/N_{}_{}_E_1.csv'.format(mu_a,sig_a),x)



