import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import math
import seaborn as sns


def upcrossings(y, upcross_level=0):
    '''
    returns mean number of upcrossings of upcross_level

    y: (n_samples, n_gridpoints) array of function values
    '''
    u = upcross_level*torch.ones(y.shape)
    return torch.mean(torch.logical_and(y[:,:-1]<u[:,:-1], y[:,1:]>u[:,1:]).type(torch.DoubleTensor), 0)

# ----------- initialization -----------

def init_bnn_from_nn(nn, bnn):
    '''
    Initializes 
    '''
    for layer_nn, layer_bnn in zip(nn.layers, bnn.layers):
        layer_bnn.w_loc.data = layer_nn.weight.data.T.detach().clone()
        layer_bnn.b_loc.data = layer_nn.bias.data.T.detach().clone()

def init_variational_parameters(shape, v0=1.0, k0=3.0, gain=1.0):
    mu0 = 0.0
    sig20 = gain**2 * 1/(shape[0]+shape[1])

    sig2 = 1/Gammma(v0/2, v0*sig20/2).sample(shape)

    normal = Normal(mu0, sig2/k0).sample(shape)
    mu = normal.sample(shape)

    return mu, sig2

def sample_normal_invgamma(mu0, v0, k0, sig20, shape):
    '''
    Samples from:
    - sig2 ~ InvGamma(v0/2, v0*sig20/2)
    - mu | sig2 ~ N(mu0, sig2/k0)
    '''
    sig2 = 1/Gamma(v0/2, v0*sig20/2).expand(shape).sample()
    mu = Normal(mu0, torch.sqrt(sig2/k0)).sample()
    return mu, sig2

def init_xavier_normal_variational_(loc, scale_untrans, untransform=lambda x: x, v0=3.0, k0=1.0, gain=1.0):
    '''
    randomly initializes location and scale parameters of variational normal distribution
    from normal-inverse-gamma distribution
    
    marginal variational distribution (i.e. marginalized over random
    location and scale parameters) is t distribution with variance
    gain**2 * (k0/(k0+1)) * ((v0-2)/v0) * 2/(Din + Dout)
    
    v0: prior samples of mean
    k0: prior samples of variance
    
    If no transformations applied to the underlying scale parameter 
    (e.g. to make it positive), set untransform to identity function (default)
    '''
    shape = loc.shape
    mu0 = 0.0
    sig20 = gain**2 * (k0/(k0+1)) * ((v0-2)/v0) * 2/sum(shape)

    mu, sig2 = sample_normal_invgamma(mu0, v0, k0, sig20, shape)

    loc.data = mu
    scale_untrans.data = untransform(sig2.sqrt())

# ----------- plotting -----------

def plot_abline(ax, slope, intercept, **plot_args):
    '''Plot a line from slope and intercept'''
    x = np.array(ax.get_xlim())
    y = intercept + slope * x
    ax.plot(x, y, **plot_args)


def make_plot_square(ax):
    '''Equalizes axes'''
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    lim = [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])]
    ax.set_xlim(lim[0], lim[1])
    ax.set_ylim(lim[0], lim[1])

def plot_predictive(x_grid, f_grid, x_train, y_train, uncertainty_type='std', f_true=None, ax=None):
    '''
    Numpy inputs

    x_grid: (n_grid, 1)
    f_true: (n_grid, 1)
    f_grid: (n_samp, n_grid)
    
    x_train: (n_train, 1)
    y_train: (n_train, 1)
    '''

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = None

    f_grid_pred = np.mean(f_grid, 0)
    ax.scatter(x_train, y_train, marker='+', color='red') # training data
    ax.plot(x_grid, f_grid_pred, color='blue', label='post mean') # predictions on grid

    n_samp_plot = min(f_grid.shape[0], 5)
    ax.plot(x_grid, f_grid[:n_samp_plot, :].T, color='blue', alpha=.1) # plot a few samples
    
    try:
        if f_true is not None:
            ax.plot(x_grid, f_true, color='orange', label='truth')
    except:
        pass

    if uncertainty_type == 'std':
        f_grid_std = np.std(f_grid, 0)

        for std in np.arange(1,3):
            f_grid_pred_lb = f_grid_pred.reshape(-1) - std * f_grid_std.reshape(-1)
            f_grid_pred_ub = f_grid_pred.reshape(-1) + std * f_grid_std.reshape(-1)

            ax.fill_between(x_grid.reshape(-1),
                             f_grid_pred_lb.reshape(-1),
                             f_grid_pred_ub.reshape(-1),
                             color='C0', alpha=0.2)

    elif uncertainty_type == 'quantile':
        for q in [2.5, 5, 10]:
            ci = np.percentile(f_grid, [q, 100-q], axis=0)

            ax.fill_between(x_grid.reshape(-1),
                             ci[0,:],
                             ci[1,:],
                             color='C0', alpha=0.2) 

    ax.legend()
    ax.set_xlabel('x')

    return fig, ax



def plot_slice(f_sampler, x, y, dim=0, n_samp=500, f_true=None, ci_type='quantile', ax=None):
    '''

    x: (N,D) training inputs
    y: (N,1) or (N,) training outputs
    quantile: Quantile of fixed x variables to use in plot
    dim: dimension of x to plot on x-axis

    Everything should be numpy
    '''

    if ax is None:
        fig, ax = plt.subplots()

    # x-axis
    midx = (x[:,dim].min() + x[:,dim].max())/2
    dx = x[:,dim].max() - x[:,dim].min()
    x_plot = np.linspace(midx - 0.75*dx, midx + 0.75*dx, 100)

    #x_plot_all = np.quantile(x, q=quantile, axis=0)*np.ones((x_plot.shape[0], x.shape[1])) # use quantile
    x_plot_all = np.zeros((x_plot.shape[0], x.shape[1])) # use zeros
    x_plot_all[:, dim] = x_plot

    # sample from model
    try:
        f_samp_plot = f_sampler(x_plot_all, n_samp) # (n_samp, N)
    except:
        f_samp_plot = np.zeros((n_samp, x_plot.shape[0]))
        for i in range(n_samp):
            f_samp_plot[i,:] = f_sampler(x_plot_all).reshape(-1)

    # plot
    ax.scatter(x[:,dim], y) # training data
    f_mean = np.mean(f_samp_plot, 0)
    ax.plot(x_plot, f_mean, color='blue', label='post mean') # posterior mean
    
    if ci_type == 'quantile':
        for q in [.025, .05, .1]:
            ci = np.quantile(f_samp_plot, [q, 1-q], axis=0)
            ax.fill_between(x_plot_all[:,dim].reshape(-1), ci[0,:], ci[1,:], alpha=.1, color='blue')
    elif ci_type == 'std':
        f_std = np.std(f_samp_plot, 0)
        for s in [1, 2]:
            ax.fill_between(x_plot_all[:,dim].reshape(-1), f_mean - s*f_std, f_mean + s*f_std, alpha=.1, color='blue')

    if f_true is not None:
        ax.plot(x_plot, f_true(x_plot_all), color='orange', label='truth') # posterior mean

    # plot a few samples
    n_samp_plot = min(10, n_samp)
    ax.plot(x_plot_all[:,dim].reshape(-1), f_samp_plot[:n_samp_plot,:].T, alpha=.1, color='blue')


def plot_slices(f_sampler, x, y, n_samp=500, f_true=None, ci_type='quantile', figsize=(4,4)):  
    dim_in = x.shape[1]  
    fig, ax = plt.subplots(1, dim_in, figsize=figsize, sharey=True)
    plt.tight_layout()

    fig.suptitle("1d slices")
    for dim in range(dim_in):
        ax_dim = ax[dim] if dim_in>1 else ax
        plot_slice(f_sampler, x, y, dim=dim, n_samp=n_samp, f_true=f_true, ci_type=ci_type, ax=ax_dim)
        ax_dim.set_xlabel('x'+str(dim))

    if dim_in>1:
        ax[0].set_ylabel('y')
        ax[0].legend()
    else:
        ax.set_ylabel('y')
        ax.legend()

    return fig, ax


def plot_upcrossings(x, f, upcross_level=0, bins=30, ax=None):
    '''
    numpy inputs
    f: (n_samp, n)
    x: (n,1) or (n,) <-- assumes this is sorted

    plots histogram of upcrossing locations
    uses midpoint of x's to assign locations (so x needs to be sorted)
    '''
    # check if x is sorted
    x = x.reshape(-1)
    x_diff = x[1:] - x[:-1]
    assert np.all(x_diff >= 0) or np.all(x_diff <= 0)
    x_mid = (x[0:-1] + x[1:])/2 # midpoint between adjacent x pairs

    u = upcross_level*np.ones(x.shape[0])
    up = np.logical_and(f[:,:-1]<u[:-1], f[:,1:]>u[1:]) # indicator of upcrossing (n_samp, n-1)

    # average upcrossings
    mean_upcrossings = np.mean(np.sum(up, 1), 0) # sum over n, average over n_samp

    # location of upcrossings
    idx_up = np.concatenate([np.where(row)[0] for row in up]) # indices of all of the upcrossings
    x_up = x_mid[idx_up] # position of all the upcrossings

    # histogram of location of upcrossings
    if ax is None:
        fig, ax = plt.subplots()
    ax.hist(x_up, bins=bins)
    ax.set_title('upcrossings --- avg = %.3f' % mean_upcrossings)

    return ax



def plot_weights_dist(w, b):
    '''
    Each input is a list over layers
    '''
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex='row', sharey='row')
    colors = plt.get_cmap("tab10")
    n_layers = len(w)-1
    assert len(w)==len(b)

    for i in range(n_layers+1):

        # w
        ax[0].set_title(r'$w$')
        sns.kdeplot(x=w[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[0])
        
        # b
        ax[1].set_title(r'$b$')
        if i < n_layers:
            sns.kdeplot(x=b[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[1])  

    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0].legend(lines, labels)

    plt.close()
    return fig, ax


def plot_param_dist(w_loc, b_loc, w_scale, b_scale):
    '''
    Each input is a list over layers
    '''
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex='row', sharey='row')
    colors = plt.get_cmap("tab10")
    n_layers = len(w_loc)-1
    assert len(w_loc)==len(b_loc)==len(w_scale)==len(b_scale)

    for i in range(n_layers+1):

        # w_loc
        ax[0,0].set_title(r'$\mu_w$')
        sns.kdeplot(x=w_loc[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[0,0])
        
        # b_loc
        ax[0,1].set_title(r'$\mu_b$')
        if i < n_layers:
            sns.kdeplot(x=b_loc[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[0,1]) 

        # w_scale
        ax[1,0].set_title(r'$\sigma_w$')
        sns.kdeplot(x=w_scale[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[1,0])
        
        # b_scale
        ax[1,1].set_title(r'$\sigma_b$')
        if i < n_layers:
            sns.kdeplot(x=b_scale[i].reshape(-1), alpha=.5, color=colors(i), fill=True, ax=ax[1,1])    

    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0,0].legend(lines, labels)

    plt.close()
    return fig, ax


def plot_weights_scatter(w0, b0, w1, b1):
    '''
    Each input is a list over layers
    '''
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex='row', sharey='row')
    colors = plt.get_cmap("tab10")
    
    assert len(w0)==len(b0)==len(w0)==len(b0)

    n_layers = len(w0)-1
    for i in range(n_layers+1):

        # w_loc
        ax[0].set_title(r'$w$')
        ax[0].scatter(w0[i], w1[i], color=colors(i))
        
        # b_loc
        ax[1].set_title(r'$b$')
        if i < n_layers:
            ax[1].scatter(b0[i], b1[i], color=colors(i))

    ax[0].set_ylabel('after training')
    ax[0].set_xlabel('before training')  
    ax[1].set_xlabel('before training')  

    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0].legend(lines, labels)

    plt.close()
    return fig, ax


def plot_param_scatter(w_loc0, b_loc0, w_scale0, b_scale0, w_loc1, b_loc1, w_scale1, b_scale1):
    '''
    Each input is a list over layers
    '''
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex='row', sharey='row')
    colors = plt.get_cmap("tab10")
    
    assert len(w_loc0)==len(b_loc0)==len(w_scale0)==len(b_scale0)
    assert len(w_loc1)==len(b_loc1)==len(w_scale1)==len(b_scale1)

    n_layers = len(w_loc0)-1
    for i in range(n_layers+1):

        # w_loc
        ax[0,0].set_title(r'$\mu_w$')
        ax[0,0].scatter(w_loc0[i], w_loc1[i], color=colors(i))
        
        # b_loc
        ax[0,1].set_title(r'$\mu_b$')
        if i < n_layers:
            ax[0,1].scatter(b_loc0[i], b_loc1[i], color=colors(i))

        # w_scale
        ax[1,0].set_title(r'$\sigma_w$')
        ax[1,0].scatter(w_scale0[i], w_scale1[i], color=colors(i)) 
        
        # b_scale
        ax[1,1].set_title(r'$\sigma_b$')
        if i < n_layers:
            ax[1,1].scatter(b_scale0[i], b_scale1[i], color=colors(i))  

    for i in range(2):
        ax[i,0].set_ylabel('after training')
        ax[1,i].set_xlabel('before training')  

    #make_plot_square(ax[0,0])
    #make_plot_square(ax[1,0])
    for a in ax.flat:
        plot_abline(a, 1, 0, color='red')

    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0,0].legend(lines, labels)

    plt.close()
    return fig, ax



def plot_weights_line(w, b, sharex=True, sharey=True, **plot_args):
    '''
    Each input is a list over layers. Can be anything corresponding to that parameter
    '''
    fig, ax = plt.subplots(1, 2, figsize=(6, 3), sharex=sharex, sharey=sharey)
    colors = plt.get_cmap("tab10")
    
    assert len(w)==len(b)

    n_layers = len(w)-1
    for i in range(n_layers+1):

        # w
        ax[0].set_title(r'$w$')
        ax[0].plot(w[i], color=colors(i), **plot_args)
        
        # b
        ax[1].set_title(r'$b$')
        if i < n_layers:
            ax[1].plot(b[i], color=colors(i), **plot_args)


    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0].legend(lines, labels)

    plt.close()
    return fig, ax


def plot_param_line(w_loc, b_loc, w_scale, b_scale, sharex=True, sharey=True, **plot_args):
    '''
    Each input is a list over layers. Can be anything corresponding to that parameter
    '''
    fig, ax = plt.subplots(2, 2, figsize=(6, 6), sharex=sharex, sharey=sharey)
    colors = plt.get_cmap("tab10")
    
    assert len(w_loc)==len(b_loc)==len(w_scale)==len(b_scale)

    n_layers = len(w_loc)-1
    for i in range(n_layers+1):

        # w_loc
        ax[0,0].set_title(r'$\mu_w$')
        ax[0,0].plot(w_loc[i], color=colors(i), **plot_args)
        
        # b_loc
        ax[0,1].set_title(r'$\mu_b$')
        if i < n_layers:
            ax[0,1].plot(b_loc[i], color=colors(i), **plot_args)

        # w_scale
        ax[1,0].set_title(r'$\sigma_w$')
        ax[1,0].plot(w_scale[i], color=colors(i), **plot_args) 
        
        # b_scale
        ax[1,1].set_title(r'$\sigma_b$')
        if i < n_layers:
            ax[1,1].plot(b_scale[i], color=colors(i), **plot_args)  

    # legend
    lines = [Line2D([0], [0], color=colors(i), linewidth=1, linestyle='-') for i in range(n_layers+1)]
    labels = ['layer %d' % i for i in range(n_layers+1)]
    ax[0,0].legend(lines, labels)

    plt.close()
    return fig, ax


def plot_elbo_loss(loss, log_prob, kl):

    fig, ax = plt.subplots(1,3, figsize=(12,3))

    if loss is not None:
        ax[0].set_title('loss (min = %.2f)' % min(loss))
        ax[0].plot(loss)

    if log_prob is not None:
        ax[1].set_title('log_prob (max = %.2f)' % max(log_prob))
        ax[1].plot(log_prob)

    if kl is not None:
        ax[2].set_title('kl (min = %.2f)' % min(kl))
        ax[2].plot(kl)

    return fig, ax 


# ----------- misc -----------

def softplus_inverse(x, beta=1.0):
    return torch.log(torch.exp(beta*x)-1.0)/beta

def kl_normal_normal_scaled(p, q, K=1):
    '''
    Decreases penalty on means by factor of 1/K
    Increases penalty on variances by factor K
    K = 1 gives regular KL 
    '''

    t1 = 1/K * ((p.loc - q.loc) / q.scale).pow(2) # decreases penalty on means
    #t1 = ((p.loc - q.loc) / q.scale).pow(2) # no scaling

    var_ratio = K * (p.scale / q.scale).pow(2) # increases penalty on variances
    ##var_ratio = (p.scale / q.scale).pow(2) # no scaling

    return 0.5 * (var_ratio + t1 - 1 - var_ratio.log())

