# standard library imports
import os
import sys
import argparse
import time

# package imports
import torch
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

import jax.numpy as jnp
import neural_tangents as nt
from neural_tangents import stax

# local imports
sys.path.append('../../..') 
import src.layers as layers
import src.util as util
import src.networks as networks
import src.callbacks as callbacks
from src.data import load_dataset

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test_mode', action='store_true') # for quickly testing figures

    # general model argument 
    parser.add_argument('--dim_in', type=int, default=1)
    parser.add_argument('--noise_sig2', type=float, default=.01, help='observational noise')
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--activation', type=str, default='erf')

    parser.add_argument('--seed_data', type=int, default=0, help='seed for dataset')

    # prior
    #parser.add_argument('--scale_by_width', action='store_true') # applies to prior and initialization
    parser.add_argument('--prior_sig2', type=float, default=1)

    # initialization
    parser.add_argument('--init_method', type=str, default='deterministic') 

    parser.add_argument('--temp_gamma_alpha', type=float, default=100.) # for initializing variational variances
    parser.add_argument('--test_param1', type=float, default=1)
    parser.add_argument('--test_param2', type=float, default=1)

    parser.add_argument('--manual_plot_lims', action='store_true')
    parser.add_argument('--dtype', type=str, default='float32')


    # gradient descent arguments
    parser.add_argument('--scale_kl', action='store_true')

    return parser 

def plot_upcrossings(x, f, upcross_level=0, bins=16, ax=None):
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

    ax.hist(x_up, bins=bins, weights=1/f.shape[0]*np.ones(len(x_up))) # weight by 1/n_samp
    #ax.set_title('upcrossings --- avg = %.3f' % mean_upcrossings)

    return ax

def plot_predictive(x_grid, f_grid, uncertainty_type='std', n_std=1, n_samp_show=0, f_true=None, color='blue', label=None, ax=None):
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
    ax.plot(x_grid, f_grid_pred, color=color, label=label) # predictions on grid

    #n_samp_plot = min(f_grid.shape[0], 5)
    if n_samp_show > 0:
        ax.plot(x_grid, f_grid[:n_samp_show, :].T, color=color, alpha=.25, linewidth=1) # plot a few samples
    
    if f_true is not None:
        ax.plot(x_grid, f_true, color='orange')

    if uncertainty_type == 'std':
        f_grid_std = np.std(f_grid, 0)

        for std in np.arange(1, n_std+1):
            f_grid_pred_lb = f_grid_pred.reshape(-1) - std * f_grid_std.reshape(-1)
            f_grid_pred_ub = f_grid_pred.reshape(-1) + std * f_grid_std.reshape(-1)

            ax.fill_between(x_grid.reshape(-1),
                             f_grid_pred_lb.reshape(-1),
                             f_grid_pred_ub.reshape(-1),
                             color=color, alpha=0.2, edgecolor='none')

    elif uncertainty_type == 'quantile':
        for q in [2.5, 5, 10]:
            ci = np.percentile(f_grid, [q, 100-q], axis=0)

            ax.fill_between(x_grid.reshape(-1),
                             ci[0,:],
                             ci[1,:],
                             color=color, alpha=0.2) 

    #ax.legend()
    #ax.set_xlabel('x')

    return fig, ax

def load_pytorch(args, dim_hidden, dir_save, act_name=None, dim_in=None):

    if act_name is None:
        act_name = args.activation

    if dim_in is None:
        dim_in = args.dim_in

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    # data
    try:
        #data = np.load(os.path.join(dir_save, 'data.npy'), allow_pickle=True).item() 
        if str(device) == 'cpu':
            # kind of a hack
            dir_save_alt = os.path.join(dir_save, '../')
            data = np.load(os.path.join(dir_save_alt, 'data.npy'), allow_pickle=True).item()
        else:
            data = np.load(os.path.join(dir_save, 'data.npy'), allow_pickle=True).item() # will this work on cuda?

        # convert to numpy
        data['x_train'] = data['x_train'].cpu().numpy()
        data['y_train'] = data['y_train'].cpu().numpy()
        data['f_train'] = data['f_train'].cpu().numpy()
        try:
            data['x_test'] = data['x_test'].cpu().numpy()
            data['y_test'] = data['y_test'].cpu().numpy()
            data['f_test'] = data['f_test'].cpu().numpy()
        except:
            pass

        #data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
        data['x_grid'] = np.linspace(-1, 1, 20).reshape(-1,1).astype(args.dtype)
    except:
        #print('unable to load data (possibly not needed)...')
        data = {'noise_sig2': args.noise_sig2}


    # model
    bnn = networks.BNN(
        dim_in=dim_in, 
        dim_hidden=dim_hidden, 
        noise_scale=np.sqrt(data['noise_sig2']), 
        n_layers=args.n_layers, 
        act_name=act_name, 
        layer_name='BBB', 
        w_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], 
        b_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], 
        ntk_scaling=True,
        temp_gamma_alpha=args.temp_gamma_alpha,
        init_method=args.init_method,
        test_param=[args.test_param1, args.test_param2],
        scale_kl = args.scale_kl)


    bnn.load_state_dict(torch.load(os.path.join(dir_save, 'model.tar'), map_location=torch.device(device)))
    bnn.to(device)
    bnn.eval()


    return bnn, data

def plot_pytorch(bnn, x_grid, prior=False, n_samp=100, n_std=1, n_samp_show=0, ax=None, label=None, color=None, ax_upcrossings=None):
    
    x_grid_torch = torch.from_numpy(x_grid).to(bnn.layers[0].device)
    bnn_grid = scalable_foward(bnn, x_grid_torch, n_samp=n_samp, prior=prior).cpu().numpy()

    plot_predictive(x_grid, bnn_grid, uncertainty_type='std', n_std=n_std, n_samp_show=n_samp_show, f_true=None, color=color, label=label, ax=ax)

    if ax_upcrossings is not None:
        plot_upcrossings(x_grid, bnn_grid, ax=ax_upcrossings)


def scalable_foward(bnn, x, n_samp=1000, prior=False, dim_hidden_thresh=256000):
    '''
    Helps with memory errors for large models 256000
    '''
    n_batch = 100 if bnn.dim_hidden < dim_hidden_thresh else n_samp # for batching over samples
    #n_batch = 10000 if bnn.dim_hidden < dim_hidden_thresh else n_samp # for batching over samples
    loop_over_data = bnn.dim_hidden >= dim_hidden_thresh  # for batching inputs
    loop_over_data=False
    with torch.no_grad():
        if loop_over_data:
            return torch.cat([bnn.forward_batched(x[i,0].reshape(1,-1), n_samp=n_samp, n_batch=n_batch, prior=prior).detach() for i in range(x.shape[0])], 0).squeeze().T # (n_samp, n_obs)
        else:
            return bnn.forward_batched(x, n_samp=n_samp, n_batch=n_batch, prior=prior).detach().squeeze().T # (n_samp, n_obs)

def forward_no_bias(bnn, x, n_samp=1000):
    '''
    Sets output bias before taking samples

    Numpy inputs/outputs, n_samp x n_obs
    '''

    # copy original q(b)
    b_loc_orig = bnn.layers[-1].b_loc.data.clone()
    b_scale_untrans_orig = bnn.layers[-1].b_scale_untrans.data.clone()

    # change zero bias
    bnn.layers[-1].b_loc.data = torch.tensor([0.], device=bnn.layers[1].device)
    bnn.layers[-1].b_scale_untrans.data = bnn.layers[-1].untransform(torch.tensor([1e-7], device=bnn.layers[1].device))

    # compute predictions
    x_torch = torch.from_numpy(x).to(bnn.layers[0].device)
    with torch.no_grad():
        #f_pred = bnn.forward_batched(x_torch, n_samp=n_samp, n_batch=100, prior=prior).detach().cpu().squeeze().numpy().T
        f_pred = scalable_foward(bnn, x_torch, n_samp=n_samp, prior=False).cpu().numpy()

    # reset to original q(b)
    bnn.layers[-1].b_loc.data = b_loc_orig
    bnn.layers[-1].b_scale_untrans.data = b_scale_untrans_orig

    return f_pred


def forward_no_bias_prior(bnn, x, n_samp=1000):
    '''
    Sets output bias before taking samples

    Numpy inputs/outputs, n_samp x n_obs

    Samples from the prior 
    '''

    # copy original prior
    p_b_orig = bnn.layers[-1].p_b

    # change to optimal bias
    bnn.layers[-1].p_b = Normal(torch.tensor([0.], device=bnn.layers[1].device), torch.tensor([1e-32], device=bnn.layers[1].device)).expand(bnn.layers[-1].b_loc.shape)

    # compute likelihood
    x_torch = torch.from_numpy(x).to(bnn.layers[0].device)
    with torch.no_grad():
        #f_pred = bnn.forward_batched(x_torch, n_samp=n_samp, n_batch=100, prior=prior).detach().cpu().squeeze().numpy().T
        f_pred = scalable_foward(bnn, x_torch, n_samp=n_samp, prior=True).cpu().numpy()

    # reset to original prior
    bnn.layers[-1].p_b = p_b_orig

    return f_pred

def compute_bound(m, bnn, data):

    assert bnn.n_layers == 1 # only works in L=1 case

    # compute optimal bias
    b_loc_opt = np.sum(data['y_train']) / (data['y_train'].shape[0] + data['noise_sig2'])
    b_scale_opt = np.sqrt(data['noise_sig2'] / (data['y_train'].shape[0] + data['noise_sig2']))

    # copy original prior
    p_b_orig = bnn.layers[-1].p_b

    # change to optimal bias
    bnn.layers[-1].p_b = Normal(torch.tensor([b_loc_opt], device=bnn.layers[1].device), torch.tensor([b_scale_opt], device=bnn.layers[1].device)).expand(bnn.layers[-1].b_loc.shape)

    # compute likelihood
    x_train_torch = torch.from_numpy(data['x_train']).to(bnn.layers[0].device)
    y_train_torch = torch.from_numpy(data['y_train']).to(bnn.layers[0].device)
    with torch.no_grad():
        f_pred = bnn.forward_batched(x_train_torch, n_samp=1000, n_batch=100, prior=True)

    log_prob = bnn.log_prob(y_train_torch, f_pred).sum()
    kl = torch.distributions.kl_divergence(bnn.layers[-1].p_b, p_b_orig).sum()

    kl_bound = (-log_prob + kl).item()

    # reset to original prior
    bnn.layers[-1].p_b = p_b_orig

    # simpler if L=D=norm_x=1, alpha=0
    c0 = 2/3 * np.sqrt(2) * kl_bound

    return c0 * m**(-0.5)

def fit_nngp(args, x_train, y_train, noise_sig2=None):
    if noise_sig2 is None:
        noise_sig2 = args.noise_sig2

    dim_hidden = 100 # why is this needed? I think it only matters if you choose to do the ntk kernel
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(dim_hidden, W_std=np.sqrt(args.prior_sig2), b_std=np.sqrt(args.prior_sig2)), stax.Erf(),
        stax.Dense(1, W_std=np.sqrt(args.prior_sig2), b_std=np.sqrt(args.prior_sig2))
    )
    predict_fn = nt.predict.gradient_descent_mse_ensemble(kernel_fn, x_train, y_train, diag_reg=np.sqrt(noise_sig2), diag_reg_absolute_scale=False) # beau: use noise level for diag_reg?
    return kernel_fn, predict_fn

def forward_nngp(mean, cov, n_samp):
    return np.random.multivariate_normal(mean.reshape(-1), cov, n_samp)

def plot_nngp(kernel_fn, predict_fn, x_grid, prior=False, n_samp=100, n_std=1, n_samp_show=0, ax=None, label=None, color=None, ax_upcrossings=None):
    if prior:
        mean = np.zeros(x_grid.shape)
        cov = kernel_fn(x_grid, x_grid, 'nngp')
    else:
        mean, cov = predict_fn(x_test=x_grid, get='nngp', compute_cov=True)

    nngp_grid = forward_nngp(mean, cov, n_samp=n_samp)
    plot_predictive(x_grid, nngp_grid, uncertainty_type='std', n_std=n_std, n_samp_show=n_samp_show, f_true=None, color=color, label=label, ax=ax)

    if ax_upcrossings is not None:
        plot_upcrossings(x_grid, nngp_grid, ax=ax_upcrossings)

def fit_nngp_alt(args, x_train, y_train, x_test, noise_sig2, prior=False):
    dim_hidden = 100 # why is this needed? I don't believe it's actually used (it shouldn't be)
    init_fn, apply_fn, kernel_fn = stax.serial(
        stax.Dense(dim_hidden, W_std=np.sqrt(args.prior_sig2), b_std=np.sqrt(args.prior_sig2)), stax.Erf(),
        stax.Dense(1, W_std=np.sqrt(args.prior_sig2), b_std=np.sqrt(args.prior_sig2))
    )
    k_xx = kernel_fn(x_train, x_train, 'nngp')

    if prior:
        mean = np.zeros(x_test.shape)
        cov = kernel_fn(x_test, x_test, 'nngp')

    else:
        k_xtest = kernel_fn(x_train, x_test, 'nngp')
        k_testtest = kernel_fn(x_test, x_test, 'nngp')
        k_xx_inv = np.linalg.inv(k_xx + noise_sig2 * np.eye(x_train.shape[0]))

        mean = k_xtest.T @ k_xx_inv @ y_train
        cov = k_testtest - k_xtest.T @ k_xx_inv @ k_xtest

    return mean, cov

def plot_nngp_alt(args, x_train, y_train, noise_sig2, x_test, prior=False, n_samp=100, n_std=1, n_samp_show=0, ax=None, label=None, color=None, ax_upcrossings=None):
    mean, cov = fit_nngp_alt(args, x_train, y_train, x_test, noise_sig2, prior=prior)
    nngp_grid = forward_nngp(mean, cov, n_samp=n_samp)
    plot_predictive(x_test, nngp_grid, uncertainty_type='std', n_std=n_std, n_samp_show=n_samp_show, f_true=None, color=color, label=label, ax=ax)

    if ax_upcrossings is not None:
        plot_upcrossings(x_test, nngp_grid, ax=ax_upcrossings)

def distance_to_prior_nngp(args, x_train, y_train, x_test, noise_sig2):
    max_error = lambda z: np.max(np.abs(z))
    
    post_mean, post_cov = fit_nngp_alt(args, x_train, y_train, x_test, noise_sig2, prior=False)
    prior_mean = 0

    error_mean = max_error(post_mean - prior_mean)
    return error_mean

def distance_to_prior(bnn, x, n_samp=1000):
    #error = lambda z: np.sqrt(np.sum(z**2))
    max_error = lambda z: np.max(np.abs(z))

    #bnn_grid_prior = forward_no_bias_prior(bnn, x, n_samp=n_samp)
    bnn_grid_post = forward_no_bias(bnn, x, n_samp=n_samp)
    
    prior_mean = 0
    #prior_var = np.var(bnn_grid_prior, 0)

    post_mean = np.mean(bnn_grid_post, 0)
    post_var = np.var(bnn_grid_post, 0)

    error_mean = max_error(post_mean - prior_mean) # Max distance
    #error_var = max_error(post_var - prior_var) # Max distance

    ### MC error
    var_est = np.sum((bnn_grid_post - post_mean.reshape(1,-1))**2, 0) / (n_samp - 1) # unbiased estimate of variance
    mc_error_mean = np.max(np.sqrt(var_est / n_samp)) # max over inputs
    ##

    return error_mean, None, mc_error_mean

def distance_to_prior_couple(bnn, x, n_samp=1000, seed=0):
    #error = lambda z: np.sqrt(np.sum(z**2))
    max_error = lambda z: np.max(np.abs(z))

    torch.manual_seed(seed)
    bnn_grid_prior = forward_no_bias_prior(bnn, x, n_samp=n_samp)

    torch.manual_seed(seed)
    bnn_grid_post = forward_no_bias(bnn, x, n_samp=n_samp)
    
    prior_mean = np.mean(bnn_grid_prior, 0)
    prior_var = np.var(bnn_grid_prior, 0)

    post_mean = np.mean(bnn_grid_post, 0)
    post_var = np.var(bnn_grid_post, 0)

    error_mean = max_error(post_mean - prior_mean) # Max distance
    error_var = max_error(post_var - prior_var) # Max distance

    ### MC error
    var_est = np.sum((bnn_grid_post - post_mean.reshape(1,-1))**2, 0) / (n_samp - 1) # unbiased estimate of variance
    mc_error_mean = np.max(np.sqrt(var_est / n_samp)) # max over inputs
    ##

    return error_mean, error_var, mc_error_mean


def distance_to_prior_rmse(bnn, x, n_samp=1000):
    #error = lambda z: np.sqrt(np.sum(z**2))
    rmse = lambda z1, z2: np.sqrt(np.mean((z1-z2)**2))

    bnn_grid_prior = forward_no_bias_prior(bnn, x, n_samp=n_samp)
    bnn_grid_post = forward_no_bias(bnn, x, n_samp=n_samp)
    
    prior_mean = 0
    prior_var = np.var(bnn_grid_prior, 0)

    post_mean = np.mean(bnn_grid_post, 0)
    post_var = np.var(bnn_grid_post, 0)

    error_mean = rmse(post_mean, prior_mean) # RMSE
    error_var = rmse(post_var, prior_var)   # RMSE

    return error_mean, error_var

def figure1(args, fontsize_titles=8, fontsize_xlabels=6, fontsize_ylabels=8, fontsize_ticks=6, n_samp=1000, n_samp_show=3):
    torch.random.manual_seed(0)
    np.random.seed(0)

    fig, ax = plt.subplots(2,2, sharex=True, sharey=True, figsize=(4,4), tight_layout=True)

    DIR_SAVE = ['../../experiment_1/results/pytorch_1/0/', '../../experiment_1/results/pytorch_1/4/']
    DIM_HIDDEN = [125, 2000]
    DIR_OUT = '../results/figure1/'

    ### for testing
    if args.test_mode:
        DIR_SAVE = ['../../experiment_1/results/pytorch_1/0/', '../../experiment_1/results/pytorch_1/1/'] # test
        DIM_HIDDEN = [125, 250] # test
    ###

    ax[0,0].set_title('Prior', fontsize=fontsize_titles)
    ax[0,1].set_title('Posterior', fontsize=fontsize_titles)

    ax[0,0].set_ylabel('BNN', fontsize=fontsize_ylabels)
    ax[1,0].set_ylabel('NNGP', fontsize=fontsize_ylabels)

    ax[1,0].set_xlabel(r'$x$', fontsize=fontsize_xlabels)
    ax[1,1].set_xlabel(r'$x$', fontsize=fontsize_xlabels)


    ## 0,0 ##
    ax_ = ax[0,0]
    bnn, data = load_pytorch(args, DIM_HIDDEN[0], DIR_SAVE[0])
    data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=True, n_samp=n_samp, n_samp_show=n_samp_show, label=r'$M=$%d' % DIM_HIDDEN[0], color='tab:blue')

    bnn, _ = load_pytorch(args, DIM_HIDDEN[1], DIR_SAVE[1])
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=True, n_samp=n_samp, n_samp_show=n_samp_show, label=r'$M=$%d' % DIM_HIDDEN[1], color='tab:orange')


    ## 0,1 ##
    ax_ = ax[0,1]
    bnn, _ = load_pytorch(args, DIM_HIDDEN[0], DIR_SAVE[0])
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, color='tab:blue')

    bnn, _ = load_pytorch(args, DIM_HIDDEN[1], DIR_SAVE[1])
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, color='tab:orange')

    ## 1,0 ##
    ax_ = ax[1,0]
    kernel_fn, predict_fn = fit_nngp(args, data['x_train'], data['y_train'], noise_sig2=bnn.noise_scale**2)
    #plot_nngp(kernel_fn, predict_fn, x_grid=data['x_grid'], prior=True, n_samp=n_samp, n_samp_show=n_samp_show, ax=ax_, label=r'$M\to\infty$', color='tab:green')
    plot_nngp_alt(args, data['x_train'], data['y_train'], noise_sig2=bnn.noise_scale**2, x_test=data['x_grid'], prior=True, n_samp=n_samp, n_samp_show=n_samp_show, ax=ax_, label=r'$M\to\infty$', color='tab:green')

    ## 1,1 ##
    ax_ = ax[1,1]
    #plot_nngp(kernel_fn, predict_fn, x_grid=data['x_grid'], prior=False, n_samp=n_samp, n_samp_show=n_samp_show, ax=ax_, label=None, color='tab:green')
    plot_nngp_alt(args, data['x_train'], data['y_train'], noise_sig2=bnn.noise_scale**2, x_test=data['x_grid'], prior=False, n_samp=n_samp, n_samp_show=n_samp_show, ax=ax_, color='tab:green')

    # adjust tick font sizes
    for i, ax_ in enumerate(ax.flat):
        ax_.tick_params(axis='both', labelsize=fontsize_ticks)
        
    # plot data over posterior plots
    for ax_ in [ax[0,1],ax[1,1]]:
        ax_.scatter(data['x_train'], data['y_train'], s=10, marker='+', linewidths=1, color='tab:red')

    # adjust xlim to remove whitespace
    ax[1,0].set_xlim(data['x_grid'].min(), data['x_grid'].max())


    legend = fig.legend(bbox_to_anchor=(.55,-.025), loc="lower center", bbox_transform=fig.transFigure, ncol=3, fontsize=6, frameon=False) # title='width'    
    plt.setp(legend.get_title(),fontsize=8)

    ax[0,0].set_ylim(-2.03,2.03)
    ax[1,0].set_ylim(-2.03,2.03)


    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    fig.savefig(os.path.join(DIR_OUT, 'figure1.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DIR_OUT, 'figure1.pdf'), dpi=300, bbox_inches='tight')



def prepare_data_mean_convergence(args, n_samp=10000, dir_out='.', fix_seed=False):
    # fix_seed: whether to use the same seed for each dim_hidden (so same seed across each seed_init)
    DIRS_DATA = ['../../experiment_3/results/pytorch_1/'] # one directory for each set of seeds
    DIM_HIDDEN_GRID = np.linspace(1e2, 1e7, 100) # for computing the bound
    MAX_DIM_HIDDEN_TEST = 1000

    rows_out = []
    for i, dir_data in enumerate(DIRS_DATA):

        arg_perms = pd.read_csv(os.path.join(dir_data, 'argument_permutations.csv'))

        for j, arg_perm in arg_perms.iterrows():

            if args.test_mode and arg_perm['--dim_hidden'] > MAX_DIM_HIDDEN_TEST:
                continue

            dir_model = os.path.join(dir_data, os.path.basename(arg_perm['--dir_out']))

            res = np.load(os.path.join(dir_model, 'results.npy'), allow_pickle=True).item()

            # grid of points
            x_grid = np.linspace(-1, 1, 25).reshape(-1,1).astype(args.dtype)

            # compute distance to the prior
            bnn, _ = load_pytorch(args, arg_perm['--dim_hidden'], dir_model, act_name=arg_perm['--activation'])

            #if fix_seed:
            #    torch.manual_seed(arg_perm['--dim_hidden']) # same seed for each dim_hidden
            error_mean, error_var, mc_error_mean = distance_to_prior_couple(bnn, x_grid, n_samp=n_samp, seed=arg_perm['--dim_hidden'])

            # compute bound based on first network (assumed to be the smallest but it doesn't matter)
            # also compute NNGP error
            if i == 0 and j==0:
                dataset = load_dataset(arg_perm['--dataset'], dim_in=bnn.dim_in, noise_sig2=bnn.noise_scale**2, n_train=arg_perm['--n_train'], n_test=100, seed=args.seed_data, dtype=args.dtype)
                data = {'x_train':dataset.x_train, 'y_train':dataset.y_train, 'noise_sig2':dataset.noise_sig2} # convert to dictionary

                bound = compute_bound(DIM_HIDDEN_GRID, bnn, data)
                error_mean_nngp = distance_to_prior_nngp(args, data['x_train'], data['y_train'], x_grid, noise_sig2=bnn.noise_scale**2)

            rows_out.append({
                'dim_hidden': arg_perm['--dim_hidden'],
                'seed': arg_perm['--seed_init'],
                'error_mean': error_mean,
                'error_var': error_var,
                'mc_error_mean': mc_error_mean,
                'bound': bound
                })

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    df = pd.DataFrame(rows_out)
    df.to_csv(os.path.join(dir_out, 'data_mean_convergence.csv'))

    df_bound = pd.DataFrame({'dim_hidden': DIM_HIDDEN_GRID, 'bound': bound, 'error_mean_nngp': error_mean_nngp})
    df_bound.to_csv(os.path.join(dir_out, 'bound_mean_convergence.csv'))
    return df, df_bound



def figure_mean_convergence(args, fontsize_titles=8, fontsize_xlabels=8, fontsize_ylabels=8, fontsize_ticks=6, n_samp=10000, fix_seed=False, fig_name='figure_mean_convergence'):
    DIR_OUT = '../results/mean_convergence'

    # obtain prepared data
    try:
        df = pd.read_csv(os.path.join(DIR_OUT, 'data_mean_convergence.csv'))
        df_bound = pd.read_csv(os.path.join(DIR_OUT, 'bound_mean_convergence.csv'))
    except:
        df, df_bound = prepare_data_mean_convergence(args, n_samp=n_samp, dir_out=DIR_OUT, fix_seed=fix_seed)

    df_agg = df.groupby('dim_hidden').agg({'error_mean':['mean','min','max']})['error_mean']

    # make plot
    fig, ax = plt.subplots(1,1, figsize=(4,2.5), tight_layout=True)
    ax.set_xscale('log', base=10)
    ax.set_yscale('log', base=10)

    # BNN
    ax.plot(df_agg.index.to_numpy(), df_agg['mean'].to_numpy(), '-o', markersize=2.5, label='Observed (BNN)') #log
    ax.fill_between(df_agg.index.to_numpy(), df_agg['min'].to_numpy(), df_agg['max'].to_numpy(), alpha=.3)
        
    # NNGP
    ax.axhline(df_bound['error_mean_nngp'][0], color='tab:green', linestyle='dashed', label='Observed (NNGP)')

    # Bound
    ax.plot(df_bound['dim_hidden'], df_bound['bound'], label='Theoretical Bound (BNN)', color='tab:orange')

    # Plot stuff
    ax.set_xlim(1e2,1e7)
    ax.set_ylabel(r'$|\mathbb{E}_{Q^*}[f(x)]| - |\mathbb{E}_{P}[f(x)]|$', fontsize=fontsize_ylabels)
    ax.set_xlabel(r'$M$', fontsize=fontsize_xlabels)

    legend = fig.legend(bbox_to_anchor=(.55,1.025), loc="upper center", bbox_transform=fig.transFigure, ncol=3, fontsize=6, frameon=False)

    ax.tick_params(axis='both', labelsize=fontsize_ticks)
    ax.tick_params(axis='x', labelsize=fontsize_ticks)

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    fig.savefig(os.path.join(DIR_OUT, '%s.png' % fig_name), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DIR_OUT, '%s.pdf' % fig_name), dpi=300, bbox_inches='tight')


def figure_counterexample(args, fontsize_titles=8, fontsize_xlabels=8, fontsize_ylabels=8, fontsize_ticks=6, n_samp=1000, n_samp_show=3):
    torch.random.manual_seed(2)
    np.random.seed(2)

    DIR_SAVE_RELU = ['../../experiment_2/results/pytorch_2/15/', '../../experiment_2/results/pytorch_1/15/'] # counterexample, 2pts
    DIR_SAVE_ODD  = ['../../experiment_1/results/pytorch_2/15/', '../../experiment_1/results/pytorch_1/15/'] # counterexample, 2pts
    DIM_HIDDEN = 4096000
    ACT_NAME = ['relu', 'erf']
    PLOT_ODD = True # whether to plot the odd activation
    DIR_OUT = '../results/counterexample'

    # For testing
    if args.test_mode:
        DIR_SAVE_RELU = ['../../experiment_2/results/pytorch_2/0/', '../../experiment_2/results/pytorch_1/0/'] # counterexample, 2pts
        DIR_SAVE_ODD  = ['../../experiment_1/results/pytorch_2/0/', '../../experiment_1/results/pytorch_1/0/'] # counterexample, 2pts
        DIM_HIDDEN = 125

    fig, ax = plt.subplots(1,2, sharex=False, sharey=False, figsize=(4,2), tight_layout=True)

    ax[0].set_title('Counterexample Dataset', fontsize=fontsize_titles)
    ax[1].set_title('Non-counterexample Dataset', fontsize=fontsize_titles)

    ax[0].set_xlabel(r'$x$', fontsize=fontsize_xlabels)
    ax[1].set_xlabel(r'$x$', fontsize=fontsize_xlabels)

    ## RELU NETWORK ## 
    ax_ = ax[0]
    bnn, data = load_pytorch(args, DIM_HIDDEN, DIR_SAVE_RELU[0], act_name=ACT_NAME[0])
    data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, label='ReLU', color='tab:blue')
    data_0 = data

    ax_ = ax[1]
    bnn, data = load_pytorch(args, DIM_HIDDEN, DIR_SAVE_RELU[1], act_name=ACT_NAME[0])
    data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
    plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, color='tab:blue')
    data_1 = data

    ## ODD NETWORK ## 
    if PLOT_ODD:
        ax_ = ax[0]
        bnn, data = load_pytorch(args, DIM_HIDDEN, DIR_SAVE_ODD[0], act_name=ACT_NAME[1])
        data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
        plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, label='erf', color='tab:orange')

        ax_ = ax[1]
        bnn, data = load_pytorch(args, DIM_HIDDEN, DIR_SAVE_ODD[1], act_name=ACT_NAME[1])
        data['x_grid'] = np.linspace(data['x_train'].min()-.5, data['x_train'].max()+.5, 50).reshape(-1,1).astype(args.dtype)
        plot_pytorch(bnn, x_grid = data['x_grid'], ax = ax_, prior=False, n_samp=n_samp, n_samp_show=n_samp_show, color='tab:orange')

    ax[0].set_xlim(data_0['x_grid'].min(), data_0['x_grid'].max())
    ax[1].set_xlim(data_1['x_grid'].min(), data_1['x_grid'].max())

    ax[0].scatter(data_0['x_train'], data_0['y_train'], s=10, marker='+', linewidths=1, color='tab:red')

    ax[1].scatter(data_1['x_train'], data_1['y_train'], s=10, marker='+', linewidths=1, color='tab:red')

    # adjust tick font sizes
    for i, ax_ in enumerate(ax.flat):
        ax_.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)
        ax_.tick_params(axis='both', which='major', labelsize=fontsize_ticks)


    if PLOT_ODD:
        fig.legend(bbox_to_anchor=(.53,0), loc="lower center", bbox_transform=fig.transFigure, ncol=3, fontsize=6, frameon=False)

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    fig.savefig(os.path.join(DIR_OUT, 'figure_counterexample.png'), dpi=300, bbox_inches='tight')
    fig.savefig(os.path.join(DIR_OUT, 'figure_counterexample.pdf'), dpi=300, bbox_inches='tight')


def prepare_data_many_datasets(args, datasets, dir_data, dir_out):
    
    NUM_SEED = 5
    MAX_DIM_HIDDEN_TEST = 500

    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    if args.test_mode:
        NUM_SEED = 2

    rows_out_all = []
    for dataset in datasets:
        rows_out = []
        for seed in range(NUM_SEED):

            # load permutations of arguments (each row is a different experiment)
            arg_perms_file = os.path.join(dir_data, '%s_%d/argument_permutations.csv' % (dataset, seed))
            try:
                arg_perms = pd.read_csv(arg_perms_file)
            except:
                print('Unable to open %s, skipping to next set of experiments' % arg_perms_file)
                continue

            for _, arg_perm in arg_perms.iterrows():

                # filepath for this experiment (row of argument permutations)
                dir_data_exp = os.path.join(dir_data, '/'.join(arg_perm['--dir_out'].split('/')[-2:]))

                if args.test_mode and arg_perm['--dim_hidden'] > MAX_DIM_HIDDEN_TEST:
                    continue

                try:
                    res = np.load(os.path.join(dir_data_exp, 'results.npy'), allow_pickle=True).item()
                except:
                    print('Unable to open %s, skipping to next experiment' % dir_save)
                    continue

                # compute distance to the prior
                bnn, data = load_pytorch(args, arg_perm['--dim_hidden'], dir_data_exp, act_name=arg_perm['--activation'], dim_in=arg_perm['--dim_in'])

                data['x_samp'] = np.random.uniform(-1,1,(100,arg_perm['--dim_in']))
                error_mean, _, _ = distance_to_prior(bnn, data['x_samp'], n_samp=1000)
                error_mean_rmse, error_var_rmse = distance_to_prior_rmse(bnn, data['x_samp'], n_samp=1000)

                # save
                rows_out.append({
                    'dataset': dataset,
                    'act': arg_perm['--activation'],
                    'seed': seed,
                    'dim_hidden': arg_perm['--dim_hidden'],
                    'max_dist_prior': error_mean,
                    'rmse_prior_mean': error_mean_rmse,
                    'rmse_prior_var': error_var_rmse,
                    'rmse_test': res['post_rmse_test']
                    })
                rows_out_all.append(rows_out[-1])
                

        df = pd.DataFrame(rows_out)
        df.to_csv(os.path.join(dir_out, 'results_%s.csv' % dataset))
    return pd.DataFrame(rows_out_all)



def figure_many_datasets(args, fontsize_titles=7, fontsize_xlabels=7, fontsize_ylabels=7, fontsize_ticks=6, n_samp=1000):
    torch.random.manual_seed(2)

    DIR_DATA = '../../experiment_4/results' # trained BNNs
    DIR_OUT = '../results/many_datasets/' # where to store the figure
    DATASETS = ['sin2', 'sin100', 'two_dim_toy100', 'concrete_slump', 'concrete_relu', 'concrete_tanh']

    if args.test_mode:
        DATASETS = ['sin100']
        
    try:
        df = pd.concat([pd.read_csv(os.path.join(DIR_OUT, 'results_%s.csv' % dataset)) for dataset in DATASETS])
        print('reading in existing prepared data')
    except:
        breakpoint()
        df = prepare_data_many_datasets(args, datasets=DATASETS, dir_data = DIR_DATA, dir_out = DIR_OUT)
    
    # collapose concrete_relu and concrete_tanh into same dataset
    df.loc[df['dataset'] == 'concrete_relu', 'dataset'] = 'concrete'
    df.loc[df['dataset'] == 'concrete_tanh', 'dataset'] = 'concrete'

    df = df.astype({'dataset': 'string','act': 'string'})

    df_long = pd.melt(df, 
                  id_vars=['dataset', 'act', 'dim_hidden'], 
                  value_vars=['rmse_prior_mean', 'rmse_test'],
                  var_name='distance_type',
                  value_name='rmse')

    # drop test rmse from sin2 (since trained on only 2 points)
    df_long = df_long.loc[(df_long['dataset'] != 'sin2') | (df_long['distance_type'] != 'rmse_test')]

    # manually change legend
    '''
    df_long.loc[df_long['dataset']=='concrete_slump', 'dataset'] = r'slump ($N=103$, $D=7$)'
    df_long.loc[df_long['dataset']=='concrete', 'dataset'] = r'concrete ($N=1030$, $D=8$)'
    df_long.loc[df_long['dataset']=='sin100', 'dataset'] = r'sine ($N=100$, $D=1$)'
    df_long.loc[df_long['dataset']=='sin2', 'dataset'] = r'2 points ($N=2$, $D=1$)'
    df_long.loc[df_long['dataset']=='two_dim_toy100', 'dataset'] = r'toy ($N=100$, $D=2$)'
    '''
    def rename_datasets(df):
        df.loc[df['dataset']=='concrete_slump', 'dataset'] = 'slump\n' + r'($N=92$, $D=7$)'
        df.loc[df['dataset']=='concrete', 'dataset'] = 'concrete\n' + r'($N=927$, $D=8$)'
        df.loc[df['dataset']=='sin100', 'dataset'] = 'sine\n' + r'($N=100$, $D=1$)'
        df.loc[df['dataset']=='sin2', 'dataset'] = '2 points\n' + r'($N=2$, $D=1$)'
        df.loc[df['dataset']=='two_dim_toy100', 'dataset'] = 'toy\n' + r'($N=100$, $D=2$)'
        return df

    def plot_rmse_mean(df_long_):
        g = sns.FacetGrid(df_long_, col="distance_type", hue="dataset", sharex=True, sharey=True, 
                  height=4)
        g.map(sns.lineplot, "dim_hidden", "rmse")
        g.tight_layout()
        g.add_legend()

        g.set(xscale='log')

        g.axes[0,0].set_xlabel(r'$M$', fontsize=fontsize_xlabels)
        g.axes[0,1].set_xlabel(r'$M$', fontsize=fontsize_xlabels)

        g.axes[0,0].set_ylabel('RMSE', fontsize=fontsize_ylabels)

        g.axes[0,0].set_title('RMSE(posterior mean, prior mean)', fontsize=fontsize_titles)
        g.axes[0,1].set_title('RMSE(posterior mean, data)', fontsize=fontsize_titles)

        #sns.move_legend(g, loc='lower center', ncol=3, bbox_to_anchor=(.40,-.18), title='')
        sns.move_legend(g, loc='lower center', ncol=3, bbox_to_anchor=(.40,-.20), title='', fontsize=6)
        g.axes[0,0].set_xlim(df['dim_hidden'].min(), df['dim_hidden'].max())


        for ax_ in g.axes.flat:
            ax_.patch.set_edgecolor('black')  
            ax_.patch.set_linewidth('1')

        for i, ax_ in enumerate(g.axes.flat):
            ax_.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)
            ax_.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

        g.fig.set_size_inches(4.25, 2.4)
        return g

    df_long = rename_datasets(df_long)
    

    ### tanh ###
    df_long_ = df_long.loc[(df_long['act'] == 'tanh')]
    g = plot_rmse_mean(df_long_)
    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_mean_tanh.png'), dpi=300, bbox_inches='tight')
    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_mean_tanh.pdf'), dpi=300, bbox_inches='tight')


    ### relu ###
    df_long_ = df_long.loc[(df_long['act'] == 'relu')]
    g = plot_rmse_mean(df_long_)
    sns.move_legend(g, loc='lower center', ncol=3, bbox_to_anchor=(.40,-.25), title='', fontsize=6)
    g.fig.set_size_inches(4.25, 2.1)
    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_mean_relu.png'), dpi=300, bbox_inches='tight')
    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_mean_relu.pdf'), dpi=300, bbox_inches='tight')


    ### variance ###
    g = sns.FacetGrid(rename_datasets(df), col="act", hue="dataset", sharex=True, sharey=True, 
                  height=4, col_order=['tanh', 'relu'])
    g.map(sns.lineplot, "dim_hidden", "rmse_prior_var")
    g.tight_layout()
    g.add_legend()


    g.set(xscale='log')
    #g.set(yscale='log')

    g.axes[0,0].set_xlabel(r'$M$', fontsize=fontsize_xlabels)
    g.axes[0,1].set_xlabel(r'$M$', fontsize=fontsize_xlabels)

    g.axes[0,0].set_ylabel('RMSE(posterior var, prior var)', fontsize=fontsize_ylabels)

    g.axes[0,0].set_title('tanh', fontsize=fontsize_titles)
    g.axes[0,1].set_title('ReLU', fontsize=fontsize_titles)

    sns.move_legend(g, loc='lower center', ncol=3, bbox_to_anchor=(.40,-.20), title='', fontsize=6)

    g.axes[0,0].set_xlim(df['dim_hidden'].min(), df['dim_hidden'].max())

    g.fig.set_size_inches(4.25, 2.5)

    for ax_ in g.axes.flat:
        ax_.patch.set_edgecolor('black')  
        ax_.patch.set_linewidth('1')

    for i, ax_ in enumerate(g.axes.flat):
            ax_.tick_params(axis='both', which='minor', labelsize=fontsize_ticks)
            ax_.tick_params(axis='both', which='major', labelsize=fontsize_ticks)

    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_var.png'), dpi=300, bbox_inches='tight')
    g.figure.savefig(os.path.join(DIR_OUT, 'figure_many_datasets_var.pdf'), dpi=300, bbox_inches='tight')
        
    






def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    figure1(args)
    figure_mean_convergence(args, n_samp=10000, fix_seed=True, fig_name='figure_mean_convergence_fixseed_T_samp_10000')
    figure_counterexample(args)
    figure_many_datasets(args)


if __name__ == '__main__':
    main()

