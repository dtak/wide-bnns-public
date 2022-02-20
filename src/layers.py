import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.gamma import Gamma
from torch.distributions import kl_divergence
import numpy as np
from math import pi, log, sqrt

import src.util as util

class BBBLayer(nn.Module):
    """
    Linear layer with Bayes by backprop
    """
    def __init__(self, dim_in, dim_out, w_loc_prior=0., b_loc_prior=0., w_scale_prior=1., b_scale_prior=1., ntk_scaling=False, temp_gamma_alpha=100., init_method='indep-normal-invgamma', test_param=None, scale_kl=False):
        super(BBBLayer, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # architecture
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.ntk_scaling = ntk_scaling
        self.temp_gamma_alpha = temp_gamma_alpha
        self.init_method = init_method
        self.test_param = test_param # for doing quick manual tests, could be used for anything
        self.scale_kl = scale_kl # whether decrease penalty on means, increase on variances

        # Variational weight (W) and bias (b) parameters
        self.w_loc = nn.Parameter(torch.empty(dim_in, dim_out))
        self.b_loc = nn.Parameter(torch.empty(dim_out))

        self.w_scale_untrans = nn.Parameter(torch.empty(dim_in, dim_out))
        self.b_scale_untrans = nn.Parameter(torch.empty(dim_out))

        # Priors
        if ntk_scaling:
            self.scale_ntk_w = w_scale_prior / sqrt(dim_in)
            self.scale_ntk_b = b_scale_prior

            w_loc_prior, w_scale_prior = 0.0, 1.0
            b_loc_prior, b_scale_prior = 0.0, 1.0

        self.register_buffer('w_loc_prior', torch.tensor([w_loc_prior]).to(self.device))
        self.register_buffer('b_loc_prior', torch.tensor([b_loc_prior]).to(self.device))
        self.register_buffer('w_scale_prior', torch.tensor([w_scale_prior]).to(self.device))
        self.register_buffer('b_scale_prior', torch.tensor([b_scale_prior]).to(self.device))

        self.p_w = Normal(self.w_loc_prior, self.w_scale_prior).expand(self.w_loc.shape)
        self.p_b = Normal(self.b_loc_prior, self.b_scale_prior).expand(self.b_loc.shape)

        self.transform = nn.Softplus() # ensure correct range for positive parameters
        self.untransform = util.softplus_inverse

        # init params either random or with pretrained net
        self.init_parameters()

    def init_parameters(self, gain=1.0):
        """
        # init means
        self.w_loc.data.normal_(0.0, 1)
        self.b_loc.data.normal_(0.0, 1)

        # init variances
        self.w_scale_untrans.data.normal_(-1, 1e-2)
        self.b_scale_untrans.data.normal_(-1, 1e-2)
        """
        #util.init_xavier_normal_variational_(self.w_loc, self.w_scale_untrans, untransform=self.untransform, gain=gain)
        #util.init_xavier_normal_variational_(self.b_loc, self.b_scale_untrans, untransform=self.untransform, gain=.1*gain)


        if self.init_method == 'indep-normal-invgamma':
            dim_hidden=self.temp_gamma_alpha # currently this is nu0

            alpha = dim_hidden + 1
            beta = dim_hidden

            #self.w_loc.data.normal_(0., sqrt(1/dim_hidden))
            self.w_loc.data.normal_(0., 1.)
            self.w_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.w_scale_untrans.shape).sample())).to(self.device)

            #self.b_loc.data.normal_(0., sqrt(1/dim_hidden))
            self.b_loc.data.normal_(0., 1.)
            self.b_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.b_scale_untrans.shape).sample())).to(self.device)


        elif self.init_method == 'deterministic':
            self.w_loc.data.zero_()
            self.w_scale_untrans.data = self.untransform(torch.ones(self.w_scale_untrans.shape))
        
            self.b_loc.data.zero_()
            self.b_scale_untrans.data = self.untransform(torch.ones(self.b_scale_untrans.shape))

        elif self.init_method == 'zero_mean':
            dim_hidden=self.temp_gamma_alpha # currently this is nu0

            alpha = dim_hidden + 1
            beta = dim_hidden

            self.w_loc.data.zero_()
            self.w_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.w_scale_untrans.shape).sample()))

            self.b_loc.data.zero_()
            self.b_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.b_scale_untrans.shape).sample()))

        elif self.init_method == 'small_std':
            self.w_loc.data.normal_(0., 1.)
            self.w_scale_untrans.data = self.untransform(1e-6 + torch.zeros(self.w_scale_untrans.shape))
        
            self.b_loc.data.normal_(0., 1.)
            self.b_scale_untrans.data = self.untransform(1e-6 +torch.zeros(self.b_scale_untrans.shape))
            

        elif self.init_method == 'test':
            '''
            layer0_std = 'large'
            layer1_std = 'small'

            if self.dim_in==1:
                if layer0_std == 'small':
                    self.w_scale_untrans.data = self.untransform(1e-8 + torch.zeros(self.w_scale_untrans.shape))
                    self.b_scale_untrans.data = self.untransform(1e-8 + torch.zeros(self.b_scale_untrans.shape))
                elif layer0_std == 'large':
                    self.w_scale_untrans.data = self.untransform(1.0 + torch.zeros(self.w_scale_untrans.shape))
                    self.b_scale_untrans.data = self.untransform(1.0 + torch.zeros(self.b_scale_untrans.shape))
            else:
                if layer1_std == 'small':
                    self.w_scale_untrans.data = self.untransform(1e-8 + torch.zeros(self.w_scale_untrans.shape))
                    self.b_scale_untrans.data = self.untransform(1e-8 + torch.zeros(self.b_scale_untrans.shape))
                elif layer1_std == 'large':
                    self.w_scale_untrans.data = self.untransform(1.0 + torch.zeros(self.w_scale_untrans.shape))
                    self.b_scale_untrans.data = self.untransform(1.0 + torch.zeros(self.b_scale_untrans.shape))

            self.w_loc.data.normal_(0., 1.)
            self.b_loc.data.normal_(0., 1.)
            '''

            # mean and variaince of inverse gamma distribution
            '''
            if self.dim_in==1:
                mean = self.test_param[0] # expectation of input layer
            else:
                mean = self.test_param[1] # expectation of output layer
            var = 1 # can manually adjust this

            # parameters of inverse gamma that'll give this mean and variance
            alpha = mean**2/var + 2
            beta = mean**3/var + mean

            self.w_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.w_scale_untrans.shape).sample()))
            self.b_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.b_scale_untrans.shape).sample()))
            
            self.w_loc.data.normal_(0., 1.)
            self.b_loc.data.normal_(0., 1.)
            '''

            nu0=5000 # currently this is nu0

            alpha = nu0 + 1
            beta = nu0

            #self.w_loc.data.normal_(0., sqrt(1/dim_hidden))
            self.w_loc.data.normal_(0., .02)
            self.w_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.w_scale_untrans.shape).sample()))

            #self.b_loc.data.normal_(0., sqrt(1/dim_hidden))
            self.b_loc.data.normal_(0., .02)
            self.b_scale_untrans.data = self.untransform(torch.sqrt(1/Gamma(alpha, beta).expand(self.b_scale_untrans.shape).sample()))



    def forward(self, x, n_samp=1, prior=False):
        '''
        x: (n_obs, dim_in) or (n_obs, n_samp, dim_in)
        '''
        if x.dim()==2:
            x = x.unsqueeze(1)

        if prior:
            dist_w, dist_b = self.p_w, self.p_b
        else:
            dist_w, dist_b = self.get_variational()
        
        if n_samp>0:
            w = dist_w.rsample((n_samp,)) # (n_samp, dim_in, dim_out)
            b = dist_b.rsample((n_samp,)) # (n_samp, dim_out)
        else:
            w = dist_w.mean.unsqueeze(0) # (1, dim_out, dim_in)
            b = dist_b.mean.unsqueeze(0) # (1, dim_out)

        if self.ntk_scaling:
            w = w * self.scale_ntk_w
            b = b * self.scale_ntk_b

        return torch.sum(x.unsqueeze(-1) * w.unsqueeze(0), 2) + b # (n_obs, n_samp, dim_out)

    def kl_divergence(self):
        """
        KL divergence (q(W) || p(W))
        """
        q_w, q_b = self.get_variational()
        if self.scale_kl:
            width = self.dim_out # proxy for width, won't work for deeper networks (i.e. only want this scaling for input layer so make self.scale_kl is False for layers except input layer)
            return util.kl_normal_normal_scaled(q_w, self.p_w, K=width).sum() + util.kl_normal_normal_scaled(q_b, self.p_b, K=width).sum()
        else:
            return kl_divergence(q_w, self.p_w).sum() + kl_divergence(q_b, self.p_b).sum()

    def get_variational(self):
        q_w = Normal(self.w_loc, self.transform(self.w_scale_untrans))
        q_b = Normal(self.b_loc, self.transform(self.b_scale_untrans))
        return q_w, q_b

    def fixed_point_updates(self):
        pass


