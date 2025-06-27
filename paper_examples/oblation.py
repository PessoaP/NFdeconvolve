# %%
# %%
import torch
from numpy import loadtxt#,sqrt,ceil,floor
from matplotlib import pyplot as plt
import nf_class
from basis import KL,logprob_gamma
from make_figures import prob_nf

from matplotlib.ticker import ScalarFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import sys
Ns = [100,316,1000]
shape, scale = 9, 1
mu_a, sig_a = 10,1
a_distribution= torch.distributions.Normal(torch.tensor(mu_a).float(),torch.tensor(sig_a).float())
filename = 'N_{}_{}_G_{}_{}.csv'.format(mu_a,sig_a,shape,scale)

# %%
#Save data, gt and the already trained standard saving the p(xb) and KL
#Verify KL match results.ipynb
x_all   = [torch.tensor(loadtxt('datasets/'+filename)[:N]).float() for N in Ns]
xb_all  = [torch.linspace((x.min().item()-(mu_a+3*sig_a)),(x.max().item()-(mu_a-3*sig_a)),10000) for x in x_all]
pgt_all = [torch.exp(logprob_gamma(xb,
                                   torch.tensor(shape,device=xb.device),
                                   torch.tensor(scale,device=xb.device))) for xb in xb_all]

default_pxb_all = [prob_nf(xb,filename,N,
                           x,a_distribution) for (x,xb,N) in zip(x_all,xb_all,Ns)]
#kl_map = KL(p_gt,p_map,xb).item()

# %%


def plot_deconvolved_pdfs(labels, pdfs_list, x_all, xb_all, pgt_all, save_name=None):
    """
    Plot histograms and deconvolved PDFs for different intervals.

    Parameters
    ----------
    labels : list of str
        Labels for each interval (M values).
    pdfs_list : list of list of arrays
        Each inner list contains arrays for each dataset, corresponding to each interval.
    x_all : list of arrays
        Raw data arrays for histograms.
    xb_all : list of arrays
        Evaluation points for PDFs.
    pgt_all : list of arrays
        Ground truth PDFs to compare against.
    save_name : str, optional
        If provided, saves the figure with the given filename and dpi=600.
    """
    fig, ax = plt.subplots(3, 4, figsize=(12, 6))

    first = True
    for (axi, pdfs) in zip(ax.T[1:], pdfs_list):
        for i in range(3):
            axi[i].plot(xb_all[i], pdfs[i].cpu(), color='r', label='NFdeconvolve' if first else None)
            axi[i].plot(xb_all[i], pgt_all[i], color='k', label='Ground truth' if first else None)
            first = False

            kl_map = KL(pgt_all[i], pdfs[i].cpu(), xb_all[i]).item()
            axi[i].text(0.95, 0.95, f"KL={kl_map:.2e}", 
                        ha='right', va='top', transform=axi[i].transAxes, 
                        fontsize=9)

    for (axi, data) in zip(ax[:, 0], x_all):
        axi.hist(data,density=True)
        axi.set_ylabel('{}  datapoints'.format(data.size(0)),fontsize=12)
    

    ax[-1][0].set_xlabel(r'$x$', fontsize=16)
    [axi.set_xlabel(r'$b$', fontsize=16) for axi in ax[-1,1:]]
    [axi.set_xlim(0, 25) for axi in ax[:,1:].reshape(-1)]
    [axi.set_xlim(ax[-1,0].get_xlim()) for axi in ax[:-1,0]]
    [axi.set_title(lab) for (axi, lab) in zip(ax[0,1:], labels)]

    plt.tight_layout()

    # Align y-axis limits across each row in ax[:,1:]
    for axi in ax[:,1:]:
        ymax = max([axij.get_ylim()[1] for axij in axi])
        [axij.set_ylim(0, ymax) for axij in axi]

    # Apply scientific notation to y-axis tick labels
    for axi in ax.reshape(-1):
        axi.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        axi.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    ax[0][0].set_title('Data', fontsize=12)
    fig.legend(loc=8, ncol=2, bbox_to_anchor=(0.5, -0.06), fontsize=12)

    if save_name:
        fig.savefig('graphs/'+save_name, dpi=600, bbox_inches='tight')



# %%
#Different layers
layers = [2,6]
pxb_layers = []
for K in layers:
    pxb =[]
    for (xd,xb,N) in zip(x_all,xb_all,Ns):
        deconvolver = nf_class.Deconvolver(xd,a_distribution,device,layers=K)
        deconvolver.train()
        pxb.append(deconvolver.get_pdf(xb)[1])
    pxb_layers.append(pxb)
pxb_layers = [pxb_layers[0],default_pxb_all,pxb_layers[1]]

labels = ['2 Layers', '4 Layers (default)', '6 Layers']
plot_deconvolved_pdfs(labels, pxb_layers, x_all, xb_all, pgt_all, save_name='layers_oblation.png')

# %%
#Different tail bounds
tails = [10,50]
pxb_tail_bounds = []
for tb in tails:
    pxb =[]
    for (xd,xb,N) in zip(x_all,xb_all,Ns):
        deconvolver = nf_class.Deconvolver(xd,a_distribution,device,tail_bound=tb)
        deconvolver.train()
        pxb.append(deconvolver.get_pdf(xb)[1])
    pxb_tail_bounds.append(pxb)
pxb_tail_bounds = [pxb_tail_bounds[0],default_pxb_all,pxb_tail_bounds[1]]


labels = ['Tail bound = 10', 'Tail bound = 30 (default)', 'Tail bound = 50']
plot_deconvolved_pdfs(labels, pxb_tail_bounds, x_all, xb_all, pgt_all, save_name='tailbound_oblation.png')

# %%
#Different intervals 
intervals = [200,2000]
pxb_intervals = []
for M in intervals:
    pxb =[]
    for (xd,xb,N) in zip(x_all,xb_all,Ns):
        deconvolver = nf_class.Deconvolver(xd,a_distribution,device,intervals=M)
        deconvolver.train()
        pxb.append(deconvolver.get_pdf(xb)[1])
    pxb_intervals.append(pxb)
pxb_intervals.append(default_pxb_all)

labels = ['M=200', 'M=2000', 'M=20000 (default)']
plot_deconvolved_pdfs(labels, pxb_intervals, x_all, xb_all, pgt_all, save_name='intervals_oblation.png')

# %%
Ns_sep = [100,100,100,316,316,1000]
Ns_control = [100, 316, 1000, 316, 1000, 1000]
unique_Ns = sorted(set(Ns_control))  # Unique and sorted sample sizes

ctr = torch.randn(size=(max(Ns_control),))*sig_a+mu_a

# Compute means only once per unique sample size
par_dict = {N: (ctr[:N].mean(),ctr[:N].std()) for N in unique_Ns}

# Reorder to match the original Ns_control
means_control = [par_dict[N][0] for N in Ns_control]
sigs_control = [par_dict[N][1] for N in Ns_control]


# %%

xb_across = xb_all[-1]
p_gt = pgt_all[-1]

pxb=[]
for i in range(len(Ns_control)):
    N,Ncontrol,mean,std = Ns_sep[i], Ns_control[i],means_control[i],sigs_control[i]
    a_d = torch.distributions.Normal(torch.tensor(mean,device=device).float(),
                                                torch.tensor(std,device=device).float())
    a_d  = a_distribution

    deconvolver = nf_class.Deconvolver(torch.tensor(loadtxt('datasets/'+filename)[:N]).float().to(device),
                                       a_d,device)
    deconvolver.train()
    pxb.append(deconvolver.get_pdf(xb_across)[1])



# %%
fig,ax = plt.subplot_mosaic([['X'    , 'c100', 'c316', 'c1000'],
                             ['d100' , 'r0'  , 'r1'  , 'r2' ],
                             ['d316' , 'X'   , 'r3'  , 'r4' ],
                             ['d1000', 'X'   , 'X'   , 'r5' ]], 
                             figsize=(12, 8), empty_sentinel='X')

[ax['d{}'.format(N)].hist(loadtxt('datasets/'+filename)[:N],density=True) for N in set(Ns_sep)]
[ax['d{}'.format(N)].set_ylabel(f'{N}  datapoints') for N in set(Ns_sep)]

[ax['c{}'.format(N)].hist(ctr[:N],color='maroon',density=True) for N in set(Ns_control)]


x_min = min(ax[f'd{N}'].get_xlim()[0] for N in set(Ns_sep))
x_max = max(ax[f'd{N}'].get_xlim()[1] for N in set(Ns_sep))
y_max = max(ax[f'd{N}'].get_ylim()[1] for N in set(Ns_sep))

for N in set(Ns_sep):
    ax[f'd{N}'].set_xlim(x_min, x_max)
    ax[f'd{N}'].set_ylim(0, y_max)

ymax = max(ax[f'c{N}'].get_ylim()[1] for N in set(Ns_control))

for N in set(Ns_control):
    ax[f'c{N}'].set_xlim((mu_a - 4 * sig_a, mu_a + 4 * sig_a))
    ax[f'c{N}'].set_ylim(0, ymax)
    mu, sigma = par_dict[N]

    ax[f'c{N}'].set_xlabel(fr'$a$')
    ax[f'c{N}'].text(
        0.95, 0.95, 
        #'test',
        fr'$\mu$ = {mu:.2f}' + '\n' + fr'$\sigma$ = {sigma:.2f}',
        ha='right', va='top',transform=ax[f'c{N}'].transAxes
    )
    ax[f'c{N}'].set_title(f'Control with {N} datapoints')




[ax['r{}'.format(r)].plot(xb_across,p_gt,color='k') for r in range(6)]
[ax['r{}'.format(r)].plot(xb_across,pxb[r].detach().cpu(),color='r') for r in range(6)]
kl_map = [KL(p_gt, pxb[r].detach().cpu(), xb_across).item() for r in range(6)]
[ax['r{}'.format(r)].text(0.95, 0.95, f"  KL={kl_map[r]:.2e}  ", 
                        ha='right', va='top', transform=ax['r{}'.format(r)].transAxes, 
                        fontsize=9) for r in range(6)]


for k in ax.keys():
    ax[k].yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax[k].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    ax[k].set_xlabel(
        ax[k].get_xlabel() if 'c' in k else
        r'$x$' if 'd' in k else
        r'$b$', 
        fontsize=16)
plt.tight_layout()


legend_elements = [
    Patch(facecolor='maroon', label='control ($a$)'),  # maroon bars
    Patch(facecolor='blue',  label='data ($x$)'),      # blue bars
    Line2D([0], [0], color='black', lw=2, label='Ground truth'),         # black line
    Line2D([0], [0], color='red', lw=2, label='NFdeconvolve')            # red line
]

fig.legend(handles=legend_elements,loc='lower center', ncol=4)
plt.savefig('graphs/wrong_adist.png',dpi=600)

# %%



