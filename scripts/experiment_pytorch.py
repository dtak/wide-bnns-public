# standard library imports
import os
import sys
import argparse
import time

# package imports
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd

# local imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 
import src.layers as layers
import src.util as util
import src.networks as networks
import src.callbacks as callbacks
from src.data import load_dataset

#torch.set_default_tensor_type(torch.float32)

def get_parser():

    parser = argparse.ArgumentParser()

    # general experiment arguments
    parser.add_argument('--dir_out', type=str, default='output/')
    parser.add_argument('--save', action='store_true', help='saves bnn object and dataset')

    # dataset arguments
    parser.add_argument('--dataset', type=str, default='sin')
    parser.add_argument('--n_train', type=int, default=10)
    parser.add_argument('--seed_data', type=int, default=0, help='seed for dataset')
    parser.add_argument('--dim_in', type=int, default=1)
    
    # general model argument 
    parser.add_argument('--noise_sig2', type=float, default=.01, help='observational noise')
    parser.add_argument('--dim_hidden', type=int, default=20)
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--activation', type=str, default='tanh')

    
    # gradient descent arguments
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_samp_elbo', type=int, default=1)
    parser.add_argument('--n_restarts', type=int, default=1)
    parser.add_argument('--seed_init', type=int, default=None) # seed for picking random seeds (for random restarts)
    parser.add_argument('--momentum', type=float, default=.9)
    parser.add_argument('--clip_grad_norm', type=float, default=None)
    parser.add_argument('--scale_kl', action='store_true')
    parser.add_argument('--batch_size', type=int, default=None)

    # prior
    #parser.add_argument('--scale_by_width', action='store_true') # applies to prior and initialization
    parser.add_argument('--prior_sig2', type=float, default=1)

    # initialization
    parser.add_argument('--pretrain_means', action='store_true')
    parser.add_argument('--init_method', type=str, default='deterministic') 
    parser.add_argument('--freeze_means_first', action='store_true')
    parser.add_argument('--freeze_scales_first', action='store_true')
    parser.add_argument('--only_train_means', action='store_true')
    parser.add_argument('--only_train_variances', action='store_true')
    parser.add_argument('--earlystopping', action='store_true')

    parser.add_argument('--n_epochs_pretrain', type=int, default=None, help='default is same as n_epochs')


    parser.add_argument('--temp_gamma_alpha', type=float, default=100.) # for initializing variational variances
    parser.add_argument('--test_param1', type=float, default=1)
    parser.add_argument('--test_param2', type=float, default=1)

    parser.add_argument('--manual_plot_lims', action='store_true')
    parser.add_argument('--dtype', type=str, default='float64')

    return parser

## Helper functions
def plot_predictive(bnn_grid, data, fname=None, metrics=None):
    '''
    bnn_grid: (n_samp, n_grid) function samples
    data: data object
    fname: filename of saved image
    metrics: dictionary of metrics to print in plot title
    '''
    if hasattr(data, 'f'):
        f_true = data.f_grid.cpu().numpy()
    else:
        f_true = None

    fig, ax = util.plot_predictive(data.x_grid.cpu().numpy(), bnn_grid.cpu().numpy(), data.x_train.cpu().numpy(), data.y_train.cpu().numpy(), f_true=f_true)
    #fig, ax = util.plot_predictive(data.x_grid, bnn_grid.cpu().numpy(), data.x_train, data.y_train, f_true=None)

    if data.y_train.shape[0] > 1:
        ax.set_ylim(data.y_train.cpu().numpy().min()-1, data.y_train.cpu().numpy().max()+1)
    else:
        ax.set_ylim(data.y_train.cpu().numpy().min()-4, data.y_train.cpu().numpy().max()+4)
    
    if metrics is not None:
        title = ' | '.join(['%s: %.3f' % (key, val) for key, val in metrics.items()])
        ax.set_title(title)
    
    if fname is not None:
        fig.savefig(fname)

def plot_predictive_slices(bnn, data, prior=False, device='cpu', fname=None):
    f_sampler = lambda x, n_samp, bnn=bnn: scalable_foward(bnn, torch.from_numpy(x).to(device), n_samp=n_samp, prior=prior).cpu().numpy()
    fig, ax = util.plot_slices(f_sampler, x=data.x_train.cpu().numpy(), y=data.y_train.cpu().numpy(), n_samp=500, f_true=None, ci_type='std', figsize=(min(data.x_train.shape[1]*3, 16),3))

    if fname is not None:
        fig.savefig(fname)

def plot_bound(bnn, data):

    from torch.distributions.normal import Normal

    ### Things to mess with
    # x
    #data.x_train = torch.tensor([-1.0, 1.0], dtype=data.x_train.dtype).reshape(-1,1)

    # y
    #data.x_train = torch.tensor([-.5, .5], dtype=data.x_train.dtype).reshape(-1,1)

    # sig2
    #data.noise_sig2 = .02 # default is .01
    #bnn.noise_scale = np.sqrt(data.noise_sig2)

    # error tolerance
    error_min = np.std(data.y_train.cpu().numpy()) / 10 # how small you want the error
    #error_min = .25
    ###

    # compute optimal bias
    b_loc_opt = np.sum(data.y_train.cpu().numpy()) / (data.y_train.shape[0] + data.noise_sig2)
    b_scale_opt = np.sqrt(data.noise_sig2 / (data.y_train.shape[0] + data.noise_sig2))

    # copy original prior
    p_b_orig = bnn.layers[-1].p_b

    # change to optimal bias
    bnn.layers[-1].p_b = Normal(b_loc_opt, b_scale_opt).expand(bnn.layers[-1].b_loc.shape)

    # compute likelihood
    with torch.no_grad():
        #f_pred = bnn.forward(data.x_train, n_samp=1000, prior=True) # (n_obs, n_samp, dim_out)
        #f_pred = torch.cat([bnn(data.x_train, n_samp=10, prior=False).detach().squeeze().T for _ in range(100)], 0) # batched
        f_pred = bnn.forward_batched(data.x_train, n_samp=1000, n_batch=100, prior=True)


    log_prob = bnn.log_prob(data.y_train, f_pred).sum()
    kl = torch.distributions.kl_divergence(bnn.layers[-1].p_b, p_b_orig).sum()

    kl_bound = (-log_prob + kl).item()

    # reset to original prior
    bnn.layers[-1].p_b = p_b_orig


    # compute bound on mean
    c1 = 8
    c2 = 1 # is this right?
    alpha = 0
    norm_x = 1
    L = bnn.n_layers
    Di = data.x_train.shape[1]

    # all the constants
    #c0 = c1 * c2**(L-1) * L * (np.abs(alpha) + 1 + norm_x/np.sqrt(Di)) * kl_bound * np.maximum((2*kl_bound)**((L-1)/2), 1)

    # simpler if L=1
    c0 = 2 / 3 * np.sqrt(1 + norm_x**2/Di) * kl_bound

    # simpler if L=D=norm_x=1, alpha=0
    c0 = 2/3 * np.sqrt(2) * kl_bound

    M_max = (c0/error_min)**2

    #breakpoint()

def scalable_foward(bnn, x, n_samp=1000, prior=False, dim_hidden_thresh=256000):
    '''
    Helps with memory errors for large models
    '''
    n_batch = 100 if bnn.dim_hidden < dim_hidden_thresh else n_samp # for batching over samples
    loop_over_data = bnn.dim_hidden >= dim_hidden_thresh  # for batching inputs

    if loop_over_data:
        return torch.cat([bnn.forward_batched(x[i,0].reshape(1,-1), n_samp=n_samp, n_batch=n_batch, prior=prior).detach() for i in range(x.shape[0])], 0).squeeze().T # (n_samp, n_obs)
    else:
        return bnn.forward_batched(x, n_samp=n_samp, n_batch=n_batch, prior=prior).detach().squeeze().T # (n_samp, n_obs)


def compute_metrics(bnn, data):
    '''
    RMSE, test log likelihood, etc
    '''
    rmse = lambda y1, y2: torch.sqrt(torch.mean((y1.reshape(-1) - y2.reshape(-1))**2)).item()
    out = {}

    post_mean_train = torch.mean(scalable_foward(bnn, data.x_train, n_samp=1000, prior=False), 0) # posterior mean, averaged over samples
    out['rmse_train'] = rmse(data.y_train, post_mean_train)

    if data.y_test is not None:
        post_mean_test = torch.mean(scalable_foward(bnn, data.x_test, n_samp=1000, prior=False), 0) # posterior mean, averaged over samples
        out['rmse_test'] = rmse(data.y_test, post_mean_test)

    return out

def save(bnn, file='bnn.tar'):
    torch.save(bnn.state_dict(),  file)

def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device %s' % device)

    # set default datatype
    if args.dtype=='float32':
        torch.set_default_dtype(torch.float32)
    elif args.dtype=='float64':
        torch.set_default_dtype(torch.float64)

    if not os.path.exists(args.dir_out):
        os.makedirs(args.dir_out)

    with open(os.path.join(args.dir_out, 'program_info.txt'), 'w') as f:
        f.write('Call:\n%s\n\n' % ' '.join(sys.argv[:]))

    if args.scale_kl:
        # kl scaling only implemented for 1 layer networks. See layers.py
        assert args.n_layers == 1

    # --------- Allocate space ----------- #
    res = {}

    n_batch = 100 if args.dim_hidden < 256000 else 1000 # for batching over samples

    # --------- Load data ----------- #
    data = load_dataset(args.dataset, dim_in=args.dim_in, noise_sig2=args.noise_sig2, n_train=args.n_train, n_test=100, seed=args.seed_data, dtype=args.dtype)

    # grid for plots (move to data.py?) 
    data.x_grid = np.linspace(data.x_train.min()-.5, data.x_train.max()+.5, 25).reshape(-1,1).astype(args.dtype)
    if hasattr(data, 'f'):
        data.f_grid = data.f(data.x_grid).astype(args.dtype)
    data.to_torch(device)
    data.x_grid = torch.from_numpy(data.x_grid).to(device)
    if hasattr(data, 'f'):
        data.f_grid = torch.from_numpy(data.f_grid).to(device)

    # --------- Define model ----------- #
    bnn = networks.BNN(
        dim_in=args.dim_in, 
        dim_hidden=args.dim_hidden, 
        noise_scale=np.sqrt(data.noise_sig2), 
        n_layers=args.n_layers, 
        act_name=args.activation, 
        layer_name='BBB', 
        w_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], 
        b_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], 
        ntk_scaling=True,
        temp_gamma_alpha=args.temp_gamma_alpha,
        init_method=args.init_method,
        test_param=[args.test_param1, args.test_param2],
        scale_kl = args.scale_kl)

    bnn.to(device)

    #plot_bound(bnn, data)

    #w_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], # used for paper
    #b_scale_prior=[np.sqrt(args.prior_sig2), np.sqrt(args.prior_sig2)], # used for paper

    #w_scale_prior=[10*np.sqrt(args.prior_sig2)]*args.n_layers+[np.sqrt(args.prior_sig2)], # experimenting
    #b_scale_prior=[10*np.sqrt(args.prior_sig2)]*args.n_layers+[np.sqrt(args.prior_sig2)], # experimenting

    # --------- Initialize parameters -----------
    if args.pretrain_means:

        ## Train NN for initialization
        net = networks.NN(
            dim_in=args.dim_in,
            dim_hidden=args.dim_hidden, 
            dim_out=1,
            n_layers=args.n_layers, 
            act_name=args.activation,
            w_init_scale=np.sqrt(args.prior_sig2),
            b_init_scale=np.sqrt(args.prior_sig2),
            ntk_scaling=True)

        net.to(device)

        printer = callbacks.Printer(frac_print=.1)
        optimizer = torch.optim.Adam(net.parameters(), lr=.01)
        tnet = networks.ModelTrainer(net)
        n_epochs_pretrain = args.n_epochs if args.n_epochs_pretrain is None else args.n_epochs_pretrain

        print('training NN...')
        history_nn = tnet.train(
            n_epochs=n_epochs_pretrain, 
            x=data.x_train, 
            y=data.y_train, 
            optimizer=optimizer, 
            callback_list=[printer])

        # Reset initialization
        def init_parameters_decorator(f):
            def inner(*args, **kwargs):
                f(*args, **kwargs)
                util.init_bnn_from_nn(net, bnn)
            return inner
        bnn.init_parameters = init_parameters_decorator(bnn.init_parameters)


    # --------- Analyze (before training) ----------- #

    # pretrained nn (if used) #
    if args.pretrain_means:
        plot_predictive(net(data.x_grid).detach().reshape(1,-1), data, os.path.join(args.dir_out, 'pretrained_nn.png'))

    # Plot prior predictive #
    with torch.no_grad():
        #bnn_grid = bnn(data.x_grid, n_samp=10000, prior=True).detach().squeeze().T
        #bnn_grid = torch.cat([bnn(data.x_grid, n_samp=10, prior=True).detach().squeeze().T for _ in range(100)], 0) # batched
        #bnn_grid = bnn.forward_batched(data.x_grid, n_samp=1000, n_batch=n_batch, prior=True).detach().squeeze().T
        
        if data.x_train.shape[1] == 1:
            bnn_grid = scalable_foward(bnn, data.x_grid, n_samp=1000, prior=True)
            plot_predictive(bnn_grid, data, os.path.join(args.dir_out, 'prior_predictives.png'))

            util.plot_upcrossings(data.x_grid.cpu().numpy(), bnn_grid.cpu().numpy())
            plt.savefig(os.path.join(args.dir_out, 'prior_upcrossings.png'))

            # stats of prior predictive #
            #res['prior_upcrossings'] = util.upcrossings(bnn_grid.cpu().numpy())
            res['prior_mean'] = torch.mean(bnn_grid, 0).cpu().numpy()
            res['prior_variance'] = torch.var(bnn_grid, 0).cpu().numpy()
        else:
            plot_predictive_slices(bnn, data, prior=True, device=device, fname=os.path.join(args.dir_out, 'prior_predictives.png'))

    # Plot initialization distribution #
    if data.x_train.shape[1] == 1:
        if args.dim_hidden < 256000:
            with torch.no_grad():
                bnn_grid = []
                for i in range(1000):
                    bnn.init_parameters(i)
                    bnn_grid.append(bnn(data.x_grid, n_samp=1, prior=False).reshape(1,-1).detach())
                bnn_grid = torch.cat(bnn_grid, dim=0)
                plot_predictive(bnn_grid, data, os.path.join(args.dir_out, 'init_predictive.png'))

    # --------- Train ----------- #

    ## Define optimization
    optimizer = torch.optim.SGD(bnn.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500)

    ## Define callbacks
    printer = callbacks.Printer(frac_print=.1, include_memory=True)
    
    #saver = callbacks.Saver(frac_start_save=.9, dir_checkpoint=args.dir_out)
    #rec = callbacks.WeightsRecorder()
    rec = None
    
    #rec_kl = callbacks.KLGradRecorder()
    rec_kl = None
    
    rec_full = callbacks.FullParamRecorder(n_record = 'all' if args.dim_hidden <= 1000 else 0)
    #rec_full = None
    
    #rec_norm = callbacks.ParamChangeNormRecorder()
    rec_norm = None

    #callback_list = [printer, rec, rec_kl, rec_full, rec_norm]
    callback_list = [printer, rec_full]

    # freezing parameteres
    if args.only_train_means:
        freezer = callbacks.ParamFreezer('scale', freeze_init=True, epochs_of_change=[None])
        callback_list.append(freezer)

    elif args.only_train_variances:
        freezer = callbacks.ParamFreezer('loc', freeze_init=True, epochs_of_change=[None])
        callback_list.append(freezer)

    elif args.freeze_means_first and args.freeze_scales_first:
        # Freeze scales (while training means), then freeze scales (training scales), then train both
        freezer_scale = callbacks.ParamFreezer('scale', freeze_init=True, epochs_of_change=[int(args.n_epochs/3)])
        freezer_loc = callbacks.ParamFreezer('loc', freeze_init=False, epochs_of_change=[int(args.n_epochs/3), 2*int(args.n_epochs/3)])

        callback_list.append(freezer_scale)
        callback_list.append(freezer_loc)

    elif args.freeze_means_first and not args.freeze_scales_first:
        freezer = callbacks.ParamFreezer('loc', freeze_init=True, epochs_of_change=[int(args.n_epochs/2)])
        callback_list.append(freezer)

    elif args.freeze_scales_first and not args.freeze_means_first:
        freezer = callbacks.ParamFreezer('scale', freeze_init=True, epochs_of_change=[int(args.n_epochs/2)])
        callback_list.append(freezer)

    if args.earlystopping:
        earlystopper = callbacks.EarlyStopper(frac_begin_lookingback=.75, frac_lookback=.25)
        callback_list.append(earlystopper)

    ## Train
    print('training BNN...')
    tbnn = networks.ModelTrainer(bnn)
    history = tbnn.train_random_restarts(
        n_restarts=args.n_restarts, 
        n_epochs=args.n_epochs, 
        x=data.x_train, 
        y=data.y_train, 
        optimizer=optimizer, 
        scheduler=scheduler,
        batch_size=args.batch_size,
        n_rep_opt=1, 
        clip_grad_norm=args.clip_grad_norm,
        callback_list=callback_list,
        seed_init=args.seed_init,
        n_samp=args.n_samp_elbo)


    # --------- Recording (after training) ----------- #
    # Note: may record things not listed here

    if args.save:
        save(bnn, os.path.join(args.dir_out, 'model.tar'))
        data.save_dict(args.dir_out)

    # Basic metrics #
    if args.dim_hidden <= 256000:
        with torch.no_grad():
            _, post_metrics = bnn.loss(data.x_train, data.y_train, n_samp=1000, n_batch=n_batch)
            for key, val in post_metrics.items():
                res['post_%s' % key] = val

            other_post_metrics = compute_metrics(bnn, data)
            for key, val in other_post_metrics.items():
                res['post_%s' % key] = val
    else:
        post_metrics = None

    # Final parameter norms #
    if rec is not None:
        res['final_norm_mu'] = rec.means_change[-1]
        res['final_norm_sig'] = rec.stds_change[-1]

        res['final_relnorm_mu'] = rec.means_relchange[-1]
        res['final_relnorm_sig'] = rec.stds_relchange[-1]

    # Final parameter norms (by layer) #
    if rec_norm is not None:
        for key, val in rec_norm.param_norms.items():
            res['final_norm_%s' % key] = val[-1].cpu().numpy()

        for key, val in rec_norm.param_relnorms.items():
            res['final_relnorm_%s' % key] = val[-1].cpu().numpy()

        for key, val in rec_norm.param_rmses.items():
            res['final_rmse_%s' % key] = val[-1].cpu().numpy()

    # --------- Plotting (after training) ----------- #

    # --- Miscellaneous plots --- #
    
    # loss #
    fig, ax = util.plot_elbo_loss(history['loss'], history['log_prob'], history['kl'])
    fig.savefig(os.path.join(args.dir_out, 'loss.png'))


    # posterior predictive #
    with torch.no_grad():
        #bnn_grid = bnn(data.x_grid, n_samp=1000, prior=False).detach().squeeze().T
        #bnn_grid = torch.cat([bnn(data.x_grid, n_samp=10, prior=False).detach().squeeze().T for _ in range(100)], 0) # batched
        #bnn_grid = bnn.forward_batched(data.x_grid, n_samp=1000, n_batch=n_batch, prior=False).detach().squeeze().T
        
        if data.x_train.shape[1] == 1:
            bnn_grid = scalable_foward(bnn, data.x_grid, n_samp=1000, prior=False)

            plot_predictive(bnn_grid, data, os.path.join(args.dir_out, 'posterior_predictive.png'), post_metrics)
            res['post_mean'] = torch.mean(bnn_grid, 0).cpu().numpy()
            res['post_var'] = torch.var(bnn_grid, 0).cpu().numpy()
        else:
            plot_predictive_slices(bnn, data, prior=False, device=device, fname=os.path.join(args.dir_out, 'posterior_predictive.png'))            

    # change in kl gradient #
    if rec_kl is not None:
        fig, ax = plt.subplots(1,2, figsize=(8,4), sharex=True)
        ax[0].plot(np.array(rec_kl.means_rel_change))
        ax[0].set_title('iterations')
        ax[0].set_title(r'relative change in $\nabla_\mu KL$ norm')

        ax[1].plot(np.array(rec_kl.stds_rel_change))
        ax[1].set_title('iterations')
        ax[1].set_title(r'relative change in $\nabla_\sigma KL$ norm')

        fig.savefig(os.path.join(args.dir_out, 'gradkl_change.png'))

    # --- Parameter norms --- #

    # norm of change #
    if rec is not None:
        fig, ax = plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True)
        ax[0].plot(np.array(rec.means_change))
        ax[0].set_title('iterations')
        ax[0].set_title(r'norm of change in $\mu$ norm')

        ax[1].plot(np.array(rec.stds_change))
        ax[1].set_title('iterations')
        ax[1].set_title(r'norm of change in $\sigma$ norm')

        ax[0].set_ylabel(r'$||\theta - \theta_0||')
        if args.manual_plot_lims:
            ax[0].set_ylim(0,4.0)

        fig.savefig(os.path.join(args.dir_out, 'param_change_norm_all.png'))

    # relative norm of change #
    if rec is not None:
        fig, ax = plt.subplots(1,2, figsize=(8,4), sharex=True, sharey=True)
        ax[0].plot(np.array(rec.means_relchange))
        ax[0].set_title('iterations')
        ax[0].set_title(r'relative norm of change in $\mu$ norm')

        ax[1].plot(np.array(rec.stds_relchange))
        ax[1].set_title('iterations')
        ax[1].set_title(r'relative norm of change in $\sigma$ norm')

        ax[0].set_ylabel(r'$||\theta - \theta_0|| / ||\theta_0||$ ')
        if args.manual_plot_lims:
            ax[0].set_ylim(0,4.0)

        fig.savefig(os.path.join(args.dir_out, 'param_change_relnorm_all.png'))


    # norm of change (by layer) #
    try:
        fig, ax = util.plot_param_line(
            w_loc = [rec_norm.param_norms['layers.%d.w_loc' % i] for i in range(args.n_layers+1)],
            b_loc = [rec_norm.param_norms['layers.%d.b_loc' % i] for i in range(args.n_layers+1)], 
            w_scale = [rec_norm.param_norms['layers.%d.w_scale_untrans' % i] for i in range(args.n_layers+1)], 
            b_scale = [rec_norm.param_norms['layers.%d.b_scale_untrans' % i] for i in range(args.n_layers+1)],
            sharex=True, sharey='row'
            )
        fig.savefig(os.path.join(args.dir_out, 'param_change_norm.png'))
    except:
        print('Unable to plot param_change_norm')

    # relative norm of change (by layer) #
    try:
        fig, ax = util.plot_param_line(
            w_loc = [rec_norm.param_relnorms['layers.%d.w_loc' % i] for i in range(args.n_layers+1)],
            b_loc = [rec_norm.param_relnorms['layers.%d.b_loc' % i] for i in range(args.n_layers+1)], 
            w_scale = [rec_norm.param_relnorms['layers.%d.w_scale_untrans' % i] for i in range(args.n_layers+1)], 
            b_scale = [rec_norm.param_relnorms['layers.%d.b_scale_untrans' % i] for i in range(args.n_layers+1)]
            )
        fig.savefig(os.path.join(args.dir_out, 'param_change_relnorm.png'))
    except:
        print('Unable to plot param_change_relnorm')

    # rmse of change (by layer) #
    try:
        fig, ax = util.plot_param_line(
            w_loc = [rec_norm.param_rmses['layers.%d.w_loc' % i] for i in range(args.n_layers+1)],
            b_loc = [rec_norm.param_rmses['layers.%d.b_loc' % i] for i in range(args.n_layers+1)], 
            w_scale = [rec_norm.param_rmses['layers.%d.w_scale_untrans' % i] for i in range(args.n_layers+1)], 
            b_scale = [rec_norm.param_rmses['layers.%d.b_scale_untrans' % i] for i in range(args.n_layers+1)],
            sharex=True, sharey='row'
            )
        fig.savefig(os.path.join(args.dir_out, 'rmse_of_param_change.png'))
    except:
        print('Unable to plot rmse_of_param_change')


    # --- Parameter values --- #
    transform = bnn.layers[0].transform # same for all layers

    # scatter plot #
    try:
        fig, ax = util.plot_param_scatter(
            w_loc0 = [rec_full.params['layers.%d.w_loc' % i][0,:] for i in range(args.n_layers+1)],
            b_loc0 = [rec_full.params['layers.%d.b_loc' % i][0,:] for i in range(args.n_layers+1)], 
            w_scale0 = [transform(rec_full.params['layers.%d.w_scale_untrans' % i][0,:]) for i in range(args.n_layers+1)], 
            b_scale0 = [transform(rec_full.params['layers.%d.b_scale_untrans' % i][0,:]) for i in range(args.n_layers+1)],
            w_loc1 = [rec_full.params['layers.%d.w_loc' % i][-1,:] for i in range(args.n_layers+1)],
            b_loc1 = [rec_full.params['layers.%d.b_loc' % i][-1,:] for i in range(args.n_layers+1)], 
            w_scale1 = [transform(rec_full.params['layers.%d.w_scale_untrans' % i][-1,:]) for i in range(args.n_layers+1)], 
            b_scale1 = [transform(rec_full.params['layers.%d.b_scale_untrans' % i][-1,:]) for i in range(args.n_layers+1)]
            )
        fig.savefig(os.path.join(args.dir_out, 'param_scatter.png'))
    except:
        print('Unable to plot param_scatter')


    # average parameter #
    try:
        fig, ax = util.plot_param_line(
            w_loc = [torch.mean(rec_full.params['layers.%d.w_loc' % i], 1) for i in range(args.n_layers+1)],
            b_loc = [torch.mean(rec_full.params['layers.%d.b_loc' % i], 1) for i in range(args.n_layers+1)], 
            w_scale = [torch.mean(transform(rec_full.params['layers.%d.w_scale_untrans' % i]), 1) for i in range(args.n_layers+1)], 
            b_scale = [torch.mean(transform(rec_full.params['layers.%d.b_scale_untrans' % i]), 1) for i in range(args.n_layers+1)],
            sharex=True, sharey='row'
            )
        fig.savefig(os.path.join(args.dir_out, 'parameters_avg.png'))
    except:
        print('Unable to plot initial parameter distribution')

   
    # density of initial parameters #
    try:
        fig, ax = util.plot_param_dist(
            w_loc = [rec_full.params['layers.%d.w_loc' % i][0,:] for i in range(args.n_layers+1)],
            b_loc = [rec_full.params['layers.%d.b_loc' % i][0,:] for i in range(args.n_layers+1)], 
            w_scale = [transform(rec_full.params['layers.%d.w_scale_untrans' % i][0,:]) for i in range(args.n_layers+1)], 
            b_scale = [transform(rec_full.params['layers.%d.b_scale_untrans' % i][0,:]) for i in range(args.n_layers+1)]
            )
        fig.savefig(os.path.join(args.dir_out, 'parameters_init.png'))
    except:
        print('Unable to plot initial parameter distribution')


    # density of final parameters #
    try:
        fig, ax = util.plot_param_dist(
            w_loc = [rec_full.params['layers.%d.w_loc' % i][-1,:] for i in range(args.n_layers+1)],
            b_loc = [rec_full.params['layers.%d.b_loc' % i][-1,:] for i in range(args.n_layers+1)], 
            w_scale = [transform(rec_full.params['layers.%d.w_scale_untrans' % i][-1,:]) for i in range(args.n_layers+1)], 
            b_scale = [transform(rec_full.params['layers.%d.b_scale_untrans' % i][-1,:]) for i in range(args.n_layers+1)]
            )
        fig.savefig(os.path.join(args.dir_out, 'parameters_final.png'))
    except:
        print('Unable to plot final parameter distribution')


    # all parameters during training #
    if args.dim_hidden < 128000:
        try:
            fig, ax = util.plot_param_line(
                w_loc = [rec_full.params['layers.%d.w_loc' % i].cpu().numpy() for i in range(args.n_layers+1)],
                b_loc = [rec_full.params['layers.%d.b_loc' % i].cpu().numpy() for i in range(args.n_layers+1)], 
                w_scale = [transform(rec_full.params['layers.%d.w_scale_untrans' % i]).cpu().numpy() for i in range(args.n_layers+1)], 
                b_scale = [transform(rec_full.params['layers.%d.b_scale_untrans' % i]).cpu().numpy() for i in range(args.n_layers+1)],
                sharex=True, sharey='row', alpha=.5
                )
            fig.savefig(os.path.join(args.dir_out, 'parameters.png'))
        except:
            print('Unable to plot all parameters')

    plt.close('all')
    np.save(os.path.join(args.dir_out, 'results.npy'), res)
    
    return res

if __name__ == '__main__':
    main()

