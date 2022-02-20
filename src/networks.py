import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import math
from math import sqrt
import os

import src.layers as layers
import src.callbacks as callbacks

def get_act(name='relu'):
    act_dict = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'rbf': lambda z: torch.exp(-0.5*z**2/1.0**2),
    'erf': torch.erf
    }
    return act_dict[name]

def get_layer(name='BBB'):
    layers_dict = {
    'BBB': layers.BBBLayer
    }
    return layers_dict[name]

class ModelTrainer(object):
    '''
    Wrapper for models

    model should have the following methods:
        loss(x, y)
        init_parameters(seed)
    '''
    def __init__(self, model):
        super(ModelTrainer).__init__()
        self.model = model

    def train_random_restarts(self, n_restarts, n_epochs, x, y, optimizer, scheduler=None, batch_size=None, n_rep_opt=1, clip_grad_norm=None, callback_list=None, seed_init=None, **kwargs_loss):
        
        # seed for random seeds
        if seed_init is not None:
            seed_gen = torch.Generator()
            seed_gen.manual_seed(seed_init)
        else:
            seed_gen = None
        seeds = torch.randint(low=0, high=100*n_restarts, size=(n_restarts,), generator=seed_gen, dtype=torch.int64)
        
        loss_best = float('inf')

        for i, seed in enumerate(seeds):
            print('random restart [%d/%d]' % (i, n_restarts))
            self.model.init_parameters(seed)
            
            history = self.train(n_epochs, x, y, optimizer, scheduler, batch_size, n_rep_opt, clip_grad_norm, callback_list, **kwargs_loss)
            
            if history['loss'][-1] < loss_best:
                # what old model was reloaded? Then history['loss'][-1] isn't the loss of the final model
                i_best = i
                seed_best = seed
                state_dict_best = self.model.state_dict().copy()
                history_best = history.copy()

        print('best was restart %d (seed=%d)' % (i_best, seed_best))
        self.model.load_state_dict(state_dict_best)

        return history_best

    def train(self, n_epochs, x, y, optimizer, scheduler=None, batch_size=None, n_rep_opt=1, clip_grad_norm=None, callback_list=None, **kwargs_loss):
        '''
        '''
        with torch.no_grad():

            # initialize history
            _, metrics_initial = self.model.loss(x, y, **kwargs_loss)
            history = {}
            for key, value in metrics_initial.items():
                history[key] = [value]
            keys_metrics = list(metrics_initial.keys())

            # set up callbacks
            if callback_list is not None:
                callback_list = callbacks.CallbackList(callback_list, self.model, optimizer)
                callback_list.on_train_begin(n_epochs, metrics_initial)

            # set up dataset
            batch_size_use = x.shape[0] if batch_size is None else batch_size
            dataloader = DataLoader(TensorDataset(x, y), batch_size = batch_size_use)


        print('training...')

        for epoch in range(1, n_epochs+1):
            
            for rep in range(n_rep_opt):

                # zero out metrics for this rep
                metrics = {}
                for key in keys_metrics:
                    metrics[key] = 0

                for batch, (x_batch, y_batch) in enumerate(dataloader):
                
                    optimizer.zero_grad()

                    loss, metrics_batch = self.model.loss(x_batch, y_batch, **kwargs_loss)

                    loss.backward()
                    if clip_grad_norm is not None:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad_norm)
                    optimizer.step()

                    # add up metrics for this batch
                    for key in keys_metrics:
                        metrics[key] += metrics_batch[key]


            # checking model (saving, early stopping, etc.)
            with torch.no_grad():

                if scheduler is not None:
                    scheduler.step(epoch)

                # update history
                [history[key].append(value) for key, value in metrics.items()]

                if callback_list is not None:
                    flags = callback_list.on_epoch_end(epoch, history)
                    if any(flags):
                        break

        with torch.no_grad():
            if callback_list is not None:
                callback_list.on_train_end(epoch, history)

        return history

class NN(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out, n_layers=1, act_name='relu', w_init_scale=1.0, b_init_scale=1.0, ntk_scaling=False):
        super(NN, self).__init__()
        '''
        Simple BNN for regression
        n_layers: number of hidden layers (must be >= 1)
        '''
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.dim_out = dim_out
        self.n_layers = n_layers
        self.w_init_scale = w_init_scale
        self.b_init_scale = b_init_scale
        self.ntk_scaling = ntk_scaling

        self.layers = nn.ModuleList(
            [nn.Linear(dim_in, dim_hidden)] + \
            [nn.Linear(dim_hidden, dim_hidden) for _ in range(n_layers-1)] + \
            [nn.Linear(dim_hidden, dim_out)] \
        )

        self.act = get_act(act_name)
        self.criterion = nn.MSELoss(reduction='sum')

        if self.ntk_scaling:
            self.scale_ntk_w = [w_init_scale*sqrt(self.dim_in)] + [sqrt(w_init_scale / self.dim_hidden)]*n_layers
            self.scale_ntk_b = [b_init_scale] + [b_init_scale]*n_layers

    def forward(self, x):
        for l, layer in enumerate(self.layers):

            if self.ntk_scaling:
                x = F.linear(x, layer.weight*self.scale_ntk_w[l], layer.bias*self.scale_ntk_b[l])
            else:
                x = layer(x)

            if l < self.n_layers:
                x = self.act(x)
        return x

    def loss(self, x, y, return_metrics=True):
        '''
        '''
        f_pred = self.forward(x)
        loss = self.criterion(f_pred, y)

        if return_metrics:
            return loss, {'loss': loss.item()}
        else:
            return loss

    def init_parameters(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)

        if self.ntk_scaling:
            for layer in self.layers:
                layer.weight.normal_(0.,1.)
                layer.bias.normal_(0.,1.)
        else:
            for layer in self.layers:
                layer.weight.normal_(0.,w_init_scale)
                layer.bias.normal_(0.,b_init_scale)
            
class BNN(nn.Module):
    """
    Simple BNN for regression (1d output)

    n_layers: number of hidden layers (must be >= 1)
    noise_scale: standard deviation of observational noise
    w_scale_prior: standard deviation of prior over weights
    b_scale_prior: standard deviation of prior over biases
    """
    def __init__(self, dim_in, dim_hidden=50, noise_scale=1., n_layers=1, act_name='relu', layer_name='BBB', w_scale_prior=1., b_scale_prior=1., temperature_kl=1.0, **kwargs_layer):
        super(BNN, self).__init__()
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.n_layers = n_layers
        self.noise_scale = noise_scale
        self.temperature_kl = temperature_kl

        self.act_name = act_name
        self.act = get_act(act_name)
        Layer = get_layer(layer_name)

        # parse prior
        if isinstance(w_scale_prior, list):
            assert len(w_scale_prior) == n_layers+1 # if input is a list, need one element for each set of weights
        else:
            w_scale_prior = [w_scale_prior]*(n_layers+1)

        if isinstance(b_scale_prior, list):
            assert len(b_scale_prior) == n_layers+1 # if input is a list, need one element for each set of weights
        else:
            b_scale_prior = [b_scale_prior]*(n_layers+1)

        # define network as list of layers
        self.layers = nn.ModuleList(
            [Layer(dim_in, dim_hidden, w_scale_prior=w_scale_prior[0], b_scale_prior=b_scale_prior[0], **kwargs_layer)] + \
            [Layer(dim_hidden, dim_hidden, w_scale_prior=w_scale_prior[i+1], b_scale_prior=w_scale_prior[i+1], **kwargs_layer) for i in range(n_layers-1)] + \
            [Layer(dim_hidden, 1, w_scale_prior=w_scale_prior[-1], b_scale_prior=w_scale_prior[-1], **kwargs_layer)] \
        )

    def forward(self, x, n_samp=1, prior=False):
        for l, layer in enumerate(self.layers):
            x = layer(x, n_samp=n_samp, prior=prior)
            if l < self.n_layers:
                x = self.act(x)
        return x

    def forward_batched(self, x, n_samp=1, n_batch=1, prior=False):
        n_samp_per_batch = int(n_samp / n_batch)
        return torch.cat([self.forward(x, n_samp=n_samp_per_batch, prior=prior) for _ in range(n_batch)], 1)

    def kl_divergence(self):
        return sum([layer.kl_divergence() for layer in self.layers])

    def log_prob(self, y_observed, f_pred):
        '''
        y_observed: (n_obs, dim_out)
        f_pred: (n_obs, n_pred, dim_out)

        averages over n_pred (e.g. could represent different samples), sums over n_obs
        '''
        lik = Normal(f_pred, self.noise_scale)
        return lik.log_prob(y_observed.unsqueeze(1)).mean(1).sum(0)

    def loss(self, x, y, return_metrics=True, n_samp=1, n_batch=1):
        '''
        uses negative elbo as loss

        n_samp: number of samples from the variational distriution for computing the likelihood term
        '''
        if n_batch == 1:
            f_pred = self.forward(x, n_samp=n_samp) # (n_obs, n_samp, dim_out)
        else:
            f_pred = self.forward_batched(x, n_samp=n_samp, n_batch=n_batch) # (n_obs, n_samp, dim_out)

        log_prob = self.log_prob(y, f_pred)
        kl = self.kl_divergence()
        loss_tempered = -log_prob + self.temperature_kl*kl # negative elbo
        if return_metrics:
            loss = -log_prob + kl # negative elbo
            metrics = {'loss': loss.item(), 'log_prob': log_prob.item(), 'kl': kl.item()}
            return loss_tempered, metrics
        else:
            return loss_tempered

    def init_parameters(self, seed=None, gain=None):
        '''
        initialize variational parameters
        '''
        if seed is not None:
            torch.manual_seed(seed)
        if gain is None:
            try:
               gain = nn.init.calculate_gain(self.act_name)
            except:
                gain = 1.0

        for layer in self.layers:
            layer.init_parameters(gain)

    def set_temperature(self, temperature):
        self.temperature_kl = temperature


