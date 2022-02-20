# standard library imports
import os
import sys

# package imports
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import autograd.numpy as np
from autograd import grad, jacobian
import torch
from scipy.stats import norm

DIR_DATA = os.path.join(os.path.dirname(__file__), '../data')

class Dataset:
    x_train = None # inputs
    f_train = None # ground truth function
    y_train = None # observed outcomes
    psi_train = None # variable importance

    x_test = None
    f_test = None
    y_test = None
    psi_test = None


class Toy(Dataset):
    def __init__(self, 
                 f,
                 x_train,
                 x_test=None,
                 noise_sig2=1.0,
                 seed=0,
                 standardize=True,
                 dtype='float64'):

        self.f = lambda x: f(x).reshape(-1,1) # makes sure output is (n,1)
        self.noise_sig2 = noise_sig2
        self.dtype = dtype

        # train
        self.x_train = x_train.astype(dtype)
        self.n_train = x_train.shape[0]
    
        # test
        self.x_test = None if x_test is None else x_test.astype(dtype)
        self.n_test = None if x_test is None else x_test.shape[0]

        self.evaluate_f()
        self.sample_y(seed)

        self.standardized = False
        if standardize:
            self.standardize()

    def evaluate_f(self):
        # train
        self.f_train = self.f(self.x_train).reshape(-1,1).astype(self.dtype)

        # test
        if self.x_test is not None:
            self.f_test = self.f(self.x_test).reshape(-1,1).astype(self.dtype)

    def sample_y(self, seed=0):
        r_noise = np.random.RandomState(seed)
        
        # train
        noise = r_noise.randn(self.n_train,1) * np.sqrt(self.noise_sig2)
        self.y_train = self.f_train + noise.astype(self.dtype)

        # test
        if self.x_test is not None:
            noise_test = r_noise.randn(self.n_test,1) * np.sqrt(self.noise_sig2)
            self.y_test = self.f_test + noise_test.astype(self.dtype)

    def standardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)
            
        if not self.standardized:
            
            self.mu_x = np.mean(self.x_train, axis=0)
            self.sigma_x = np.std(self.x_train, axis=0)

            #self.mu_f = np.mean(self.f_train, axis=0)
            #self.sigma_f = np.std(self.f_train, axis=0)

            self.mu_y = np.mean(self.y_train, axis=0)
            self.sigma_y = np.std(self.y_train, axis=0)

            self.x_train = zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f_orig = self.f
            self.f = lambda x: zscore(self.f_orig(un_zscore(x, self.mu_x, self.sigma_x)), self.mu_y, self.sigma_y)
            self.standardized = True

    def unstandardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)    

        if self.standardized:
            
            self.x_train = un_zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = un_zscore(self.x_test, self.mu_x, self.sigma_x)

            self.f_train = un_zscore(self.f_train, self.mu_y, self.sigma_y)
            if self.f_test is not None:
                self.f_test = un_zscore(self.f_test, self.mu_y, self.sigma_y)

            self.y_train = un_zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = un_zscore(self.y_test, self.mu_y, self.sigma_y)

            self.f = self.f_orig
            self.standardized = False

    def to_torch(self, device='cpu'):
        self.x_train = torch.from_numpy(self.x_train).to(device)
        self.f_train = torch.from_numpy(self.f_train).to(device)
        self.y_train = torch.from_numpy(self.y_train).to(device)

        if self.x_test is not None:
            self.x_test = torch.from_numpy(self.x_test).to(device)
            self.f_test = torch.from_numpy(self.f_test).to(device)
            self.y_test = torch.from_numpy(self.y_test).to(device)

    def to_numpy(self):
        self.x_train = self.x_train.numpy()
        self.f_train = self.f_train.numpy()
        self.y_train = self.y_train.numpy()

        if self.x_test is not None:
            self.x_test = self.x_test.numpy()
            self.f_test = self.f_test.numpy()
            self.y_test = self.y_test.numpy()

    def save_dict(self, dir):
        np.save(
            os.path.join(dir, 'data.npy'),
            {'x_train': self.x_train,
            'f_train': self.f_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'f_test': self.f_test,
            'y_test': self.y_test,
            'noise_sig2': self.noise_sig2})


class RealToy(Dataset):
    '''
    Either input x, y and split into train/test randomly OR input train/test manually
    '''
    def __init__(self, 
                 x=None,
                 y=None,
                 x_train=None, 
                 y_train=None,
                 x_test=None,
                 y_test=None,
                 noise_sig2=1.0,
                 frac_split=.8,
                 seed_split=0,
                 standardize=True,
                 dtype='float64'):

        self.noise_sig2 = noise_sig2 # not used
        self.dtype = dtype

        if x is not None and y is not None:
            self.x = x.astype(dtype)
            self.y = y.astype(dtype)

            self.n = x.shape[0]
            self.n_train = int(frac_split * self.n)
            self.n_test = self.n - self.n_train

            self.split(seed = seed_split)

        else:
            self.x_train = x_train.astype(dtype)
            self.x_test = x_test.astype(dtype)
            self.y_train = y_train.astype(dtype)
            self.y_test = y_test.astype(dtype)
        

        self.standardized = False
        if standardize:
            self.standardize()
        

    def split(self, seed=0):
        # train/test split

        r = np.random.RandomState(seed)
        idx_train = r.choice(self.n, self.n_train, replace=False)
        idx_test = np.setdiff1d(np.arange(self.n), idx_train)

        self.x_train = self.x[idx_train, :]
        self.x_test = self.x[idx_test, :]

        self.y_train = self.y[idx_train].reshape(-1, 1)
        self.y_test = self.y[idx_test].reshape(-1, 1)

    def standardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)
            
        if not self.standardized:
            
            self.mu_x = np.mean(self.x_train, axis=0)
            self.sigma_x = np.std(self.x_train, axis=0)

            self.mu_y = np.mean(self.y_train, axis=0)
            self.sigma_y = np.std(self.y_train, axis=0)

            self.x_train = zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = zscore(self.x_test, self.mu_x, self.sigma_x)

            self.y_train = zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = zscore(self.y_test, self.mu_y, self.sigma_y)

            self.standardized = True

    def unstandardize(self):
        zscore = lambda x, mu, sigma: (x - mu.reshape(1,-1)) / sigma.reshape(1,-1)
        un_zscore = lambda x, mu, sigma: x * sigma.reshape(1,-1) + mu.reshape(1,-1)    

        if self.standardized:
            
            self.x_train = un_zscore(self.x_train, self.mu_x, self.sigma_x)
            if self.x_test is not None:
                self.x_test = un_zscore(self.x_test, self.mu_x, self.sigma_x)

            self.y_train = un_zscore(self.y_train, self.mu_y, self.sigma_y)
            if self.y_test is not None:
                self.y_test = un_zscore(self.y_test, self.mu_y, self.sigma_y)

            self.standardized = False

    def to_torch(self, device='cpu'):
        self.x_train = torch.from_numpy(self.x_train).to(device)
        self.y_train = torch.from_numpy(self.y_train).to(device)

        if self.x_test is not None:
            self.x_test = torch.from_numpy(self.x_test).to(device)
            self.y_test = torch.from_numpy(self.y_test).to(device)

    def to_numpy(self):
        self.x_train = self.x_train.numpy()
        self.y_train = self.y_train.numpy()

        if self.x_test is not None:
            self.x_test = self.x_test.numpy()
            self.y_test = self.y_test.numpy()

    def save_dict(self, dir):
        np.save(
            os.path.join(dir, 'data.npy'),
            {'x_train': self.x_train,
            'y_train': self.y_train,
            'x_test': self.x_test,
            'y_test': self.y_test,
            'noise_sig2': self.noise_sig2})


def sin_toy(dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, dtype='float64'):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.uniform(-5,5,(n_train, dim_in))
    x_test = r_x.uniform(-5,5,(n_test, dim_in))

    # ground truth function
    f = lambda x: np.sum(np.sin(x),-1).reshape(-1,1)

    return Toy(f, x_train, x_test, noise_sig2, seed_noise, dtype=dtype)

def sin0_toy(dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, dtype='float64'):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.uniform(-5,5,(n_train, dim_in))
    x_test = r_x.uniform(-5,5,(n_test, dim_in))

    # ground truth function
    f = lambda x: np.sin(x[:,0]).reshape(-1,1) # only uses first dimension

    return Toy(f, x_train, x_test, noise_sig2, seed_noise, dtype=dtype)


def two_dim_toy(dim_in, noise_sig2, n_train, n_test=100, seed_x=0, seed_noise=0, dtype='float64'):
    assert dim_in==2

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.uniform(-5,5,(n_train, dim_in))
    x_test = r_x.uniform(-5,5,(n_test, dim_in))

    # ground truth function
    f = lambda x: x[:,0].reshape(-1,1) * np.sin(x[:,1]).reshape(-1,1)

    return Toy(f, x_train, x_test, noise_sig2, seed_noise, dtype=dtype)


def rff_toy(dim_in, noise_sig2, n_train, n_test=100, dim_hidden=50, seed_x=0, seed_w=0, seed_noise=0, dtype='float64'):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.randn(n_train, dim_in)
    x_test = r_x.randn(n_test, dim_in)

    # ground truth function
    r_w = np.random.RandomState(seed_w)
    w1 = r_w.randn(dim_hidden, dim_in)
    b1 = r_w.uniform(0, 2*np.pi, (dim_hidden,1))
    w2 = r_w.randn(1, dim_hidden)
    act = lambda z: np.sqrt(2/dim_hidden)*np.cos(z)
    f = lambda z: act(z@w1.T + b1.T) @ w2.T

    return Toy(f, x_train, x_test, noise_sig2, seed_noise, dtype=dtype)


def blog_toy(dtype='float64'):
    '''
    two observation dataset from blog post
    '''
    dataset = Toy(f = lambda x: -6/7*x - 4/7, x_train=np.array([-3, 0.5]).reshape(-1,1), x_test=None, noise_sig2=0., standardize=False, dtype=dtype)
    dataset.noise_sig2 = 1e-4 # overwrite noise variance (so true data sampled with no noise)
    return dataset

def origin_toy(dim_in, n_train=1, dtype='float64'):
    '''
    x=0, y=0
    '''
    x_train = np.zeros((n_train, dim_in))
    f = lambda x: x @ np.zeros((x.shape[1], 1))
    dataset = Toy(f, x_train, x_test=None, noise_sig2=0., standardize=False, dtype=dtype)
    dataset.noise_sig2 = 1e-4 # overwrite noise variance (so true data sampled with no noise)
    return dataset

def zero_toy(dim_in, n_train, n_test=100, seed_x=0, dtype='float64'):

    # sample x
    r_x = np.random.RandomState(seed_x)
    x_train = r_x.uniform(-1,1,size=(n_train,dim_in))
    x_test = r_x.uniform(-1,1,size=(n_test,dim_in))

    f = lambda x: x @ np.zeros((x.shape[1], 1))
    dataset = Toy(f, x_train, x_test=x_test, noise_sig2=0., standardize=False, dtype=dtype)
    dataset.noise_sig2 = 1e-4 # overwrite noise variance (so true data sampled with no noise)
    
    return dataset

def above_zero_toy(dim_in, n_train, n_test=100, seed_x=0, dtype='float64'):
    '''
    f=0 but higher concentration of x's above zero
    '''
    r_x = np.random.RandomState(seed_x)

    def sample_x(n):
        n_below = 1*int(n/5)
        n_above = 4*int(n/5) + n%5

        x_below = r_x.uniform(-1.5,-0.5,size=(n_below, dim_in))
        x_above = r_x.uniform(0.5,1.5,size=(n_above, dim_in))

        x = np.concatenate((x_below, x_above), axis=0)
        np.random.shuffle(x)
        return x

    # sample x
    
    x_train = sample_x(n_train)
    x_test = sample_x(n_test)

    f = lambda x: x @ np.zeros((x.shape[1], 1))
    dataset = Toy(f, x_train, x_test=x_test, noise_sig2=0., standardize=False, dtype=dtype)
    dataset.noise_sig2 = 1e-4 # overwrite noise variance (so true data sampled with no noise)
    
    return dataset

'''
def pods(dtype='float64'):

    """
    import pods
    data = pods.datasets.olympic_marathon_men()
    x_train = data['X'].reshape(-1,1)
    y_train = data['Y'].reshape(-1,1)
    """

    data = pd.read_csv(os.path.join(DIR_DATA, 'marathon/train.csv'))
    x_train = data['x_train'].to_numpy().reshape(-1,1)
    y_train = data['y_train'].to_numpy().reshape(-1,1)

    f = lambda x, y_train=y_train: y_train

    dataset = Toy(f, x_train, x_test=None, noise_sig2=0., standardize=True, dtype=dtype)
    dataset.noise_sig2 = .1 # overwrite noise variance (so true data sampled with no noise)
    return dataset
'''

def pods(dtype='float64'):
    """
    import pods
    data = pods.datasets.olympic_marathon_men()
    x_train = data['X'].reshape(-1,1)
    y_train = data['Y'].reshape(-1,1)
    """

    data = pd.read_csv(os.path.join(DIR_DATA, 'marathon/train.csv'))

    x = data['x_train'].to_numpy().reshape(-1,1)
    y = data['y_train'].to_numpy().reshape(-1,1)

    dataset = RealToy(x, y, noise_sig2=.025, frac_split=.9, seed_split=0, standardize=True, dtype='float64')
    return dataset


def yacht(seed_split = 0, dtype='float64'):
    data = pd.read_csv(os.path.join(DIR_DATA, 'yacht/yacht_hydrodynamics.data'), delimiter='\s+', header=None)

    x = data.to_numpy()[:,:6]
    y = data.to_numpy()[:,6]

    dataset = RealToy(x, y, noise_sig2=.025, frac_split=.9, seed_split=seed_split, standardize=True, dtype='float64')
    return dataset

def concrete(seed_split = 0, dtype='float64'):
    data = pd.read_csv(os.path.join(DIR_DATA, 'concrete/Concrete_Data.csv'), delimiter=',', header=0)

    x = data.to_numpy()[:,:8]
    y = data.to_numpy()[:,8]

    dataset = RealToy(x, y, noise_sig2=.025, frac_split=.9, seed_split=seed_split, standardize=True, dtype='float64')
    return dataset

def concrete_slump(seed_split = 0, dtype='float64'):
    data = pd.read_csv(os.path.join(DIR_DATA, 'concrete_slump/slump_test.data'), delimiter=',', header=0)

    x = data.to_numpy()[:,1:8]
    y = data.to_numpy()[:,8]

    dataset = RealToy(x, y, noise_sig2=.025, frac_split=.9, seed_split=seed_split, standardize=True, dtype='float64')
    return dataset

def counterexample(dim_in, n_train, seed_x=0, dtype='float64'):
    '''
    Countexample dataset for single layer (L=1)
    '''

    assert dim_in==1
    assert n_train==2

    # Define NNGP kernel for ReLU
    def k_nngp(x, L=1, sig_w=1.0, sig_b=1.0):
        return sig_b**2 * np.sum((sig_w**2/2)**np.arange(L+1)) + (sig_w**2/2)**(L+1) * x**2
        
    # Define lam(x) = E[ReLU(z(x))]
    def lam(x, mu_w=0.0, sig_w=1.0, mu_b=0.0, sig_b=1.0):

        # z = wx+b
        mu_z = x * mu_w + mu_b
        sig_z = np.sqrt(x**2 * sig_w**2 + sig_b**2)

        # E[ReLU(z)]
        return mu_z * (1-norm.cdf(-mu_z/sig_z)) + sig_z * norm.pdf(-mu_z/sig_z)

    # Find x, x' such that lam(x) != lam(x')
    x = 0
    xp = 1.0
    assert lam(x) != lam(xp)

    # pick beta such that |lam(x) - lam(x')| >= beta/sqrt(2)
    beta = np.sqrt(2) * np.abs(lam(x) - lam(xp)) * .9

    # pick C such that sqrt(C)*beta/sqrt(2) - sqrt(2)sqrt(1+k(x)+k(x')+beta**2)
    C_min = 4/beta**2 * (1 + k_nngp(x) + k_nngp(xp) + beta**2)
    C = 1.1 * C_min

    y = np.sqrt(C) * lam(x)
    yp = np.sqrt(C) * lam(xp)

    noise_sig2 = 1/C # observational noise

    # create dataset object
    x_train = np.array([x, xp]).reshape(-1,1)
    y_train = np.array([y, yp]).reshape(-1,1)
    f = lambda x, y_train=y_train: y_train

    dataset = Toy(f, x_train, x_test=None, noise_sig2=0., standardize=False, dtype=dtype)
    dataset.noise_sig2 = noise_sig2
    return dataset


def load_dataset(name, dim_in, noise_sig2, n_train=100, n_test=100, signal_scale=1.0, seed=0, dtype='float64'):
    '''
    inputs:

    returns:
    '''

    if name == 'sin':
        dataset = sin_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=seed, seed_noise=seed, dtype=dtype)

    if name == 'sin0':
        dataset = sin0_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=0, seed_noise=0, dtype=dtype)

    if name == 'two_dim':
        dataset = two_dim_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=0, seed_noise=0, dtype=dtype)

    elif name == 'rff':
        dataset = rff_toy(dim_in, noise_sig2, n_train, n_test=n_test, seed_x=0, seed_noise=0, dtype=dtype)

    elif name == 'blog':
        dataset = blog_toy(dtype=dtype)

    elif name == 'origin':
        dataset = origin_toy(dim_in=dim_in, dtype=dtype)

    elif name == 'zero':
        dataset = zero_toy(dim_in=dim_in, n_train=n_train, dtype=dtype)

    elif name == 'above_zero':
        dataset = above_zero_toy(dim_in=dim_in, n_train=n_train, dtype=dtype)

    elif name == 'pods':
        assert dim_in == 1
        dataset = pods(dtype=dtype)
        dataset.noise_sig2 = noise_sig2

    elif name == 'yacht':
        assert dim_in == 6
        dataset = yacht(seed_split=seed, dtype=dtype)
        dataset.noise_sig2 = noise_sig2

    elif name == 'concrete':
        assert dim_in == 8
        dataset = concrete(seed_split=seed, dtype=dtype)
        dataset.noise_sig2 = noise_sig2

    elif name == 'concrete_slump':
        assert dim_in == 7
        dataset = concrete_slump(seed_split=seed, dtype=dtype)
        dataset.noise_sig2 = noise_sig2

    elif name == 'counterexample':
        dataset = counterexample(dim_in=dim_in, n_train=n_train, dtype=dtype)

    return dataset
    
if __name__ == "__main__":

    dir_out = './datasets'
    if not os.path.exists(dir_out):
        os.makedirs(dir_out)

    names = ['sin', 'rff']
    dim_in = 3

    for name in names:

        try:
            data = load_dataset(name, dim_in=dim_in, noise_sig2=.01, n_train=100, n_test=100)
        except:
            print('Unable to load dataset %s' % name)
            continue

        data.unstandardize()
        
        # variable importance
        if data.psi_train is not None:
            fig, ax = plt.subplots()
            variable = np.arange(dim_in).tolist()
            psi = data.psi_train.tolist()
            split = ['train']*dim_in

            if data.psi_test is not None:
                variable += np.arange(dim_in).tolist()
                psi += data.psi_test.tolist()
                split += ['test']*dim_in

            psi_df = pd.DataFrame({
                'variable': variable,
                'psi': psi,
                'split': split
                })
            fig, ax = plt.subplots()
            ax.set_title(r'variable importance $\psi$')
            sns.barplot(x="variable", y="psi", hue="split", data=psi_df, ax=ax)
            fig.savefig(os.path.join(dir_out, 'dataset=%s_var_import.png' % name))
            plt.close()

        # data
        fig, ax = plt.subplots(1,dim_in, figsize=(12,4))
        for i in range(dim_in):
            # train
            ax[i].scatter(data.x_train[:,i], data.y_train, label='train', color='blue')
            
            # test
            if data.x_test is not None and data.y_test is not None:
                ax[i].scatter(data.x_test[:,i], data.y_test, label='test', color='red')            

            if hasattr(data, 'f'):
                n_grid = 100
                #x_grid = np.ones((n_grid,dim_in)) * np.median(data.x_train,0).reshape(1,-1) # hold other variables at median
                x_grid = np.zeros((n_grid,dim_in)) # hold other variables at zero


                x_grid_i = np.linspace(
                    min(data.x_train[:,i].min(), data.x_test[:,i].min()),
                    max(data.x_train[:,i].max(), data.x_test[:,i].max()),
                    n_grid
                    )
                x_grid[:,i] = x_grid_i
                f_grid_i = data.f(x_grid)
                ax[i].plot(x_grid_i, f_grid_i, label='f', color='black')
                
        ax[0].legend()
    
        fig.savefig(os.path.join(dir_out, 'dataset=%s.png' % name))





