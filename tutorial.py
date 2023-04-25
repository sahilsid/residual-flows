import time
import math
import os
import os.path
import numpy as np
from tqdm import tqdm
import gc
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
import torchvision.datasets as vdsets
from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt  

class Config:
  data        = "celebahq"
  imagesize   = 256
  nbits       = 5
  block       = "resblock"
  coeff       = 0.98
  vnorms      = '2222'
  batchsize   = 1
  nblocks     = "16-16-16-16-16-16"
  act         = "elu"
  factor_out  = True
  padding     = 0
  squeeze_first = True
  resume = "celebahq256_resflow_16-16-16-16-16-16.pth"

args = Config()

# Random seed
args.seed = np.random.randint(100000)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if device.type == 'cuda':
    torch.cuda.manual_seed(args.seed)

def geometric_logprob(ns, p):
    return torch.log(1 - p + 1e-10) * (ns - 1) + torch.log(p + 1e-10)


def standard_normal_sample(size):
    return torch.randn(size)


def standard_normal_logprob(z):
    logZ = -0.5 * math.log(2 * math.pi)
    return logZ - z.pow(2) / 2

def normal_logprob(z, mean, log_std):
    mean = mean + torch.tensor(0.)
    log_std = log_std + torch.tensor(0.)
    c = torch.tensor([math.log(2 * math.pi)]).to(z)
    inv_sigma = torch.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def reduce_bits(x):
    if args.nbits < 8:
        x = x * 255
        x = torch.floor(x / 2**(8 - args.nbits))
        x = x / 2**args.nbits
    return x

def add_noise(x, nvals=256):
    """
    [0, 1] -> [0, nvals] -> add noise -> [0, 1]
    """
    if args.add_noise:
        noise = x.new().resize_as_(x).uniform_()
        x = x * (nvals - 1) + noise
        x = x / nvals
    return x

def add_padding(x, nvals=256):
    # Theoretically, padding should've been added before the add_noise preprocessing.
    # nvals takes into account the preprocessing before padding is added.
    if args.padding > 0:
        if args.padding_dist == 'uniform':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).uniform_()
            logpu = torch.zeros_like(u).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        elif args.padding_dist == 'gaussian':
            u = x.new_empty(x.shape[0], args.padding, x.shape[2], x.shape[3]).normal_(nvals / 2, nvals / 8)
            logpu = normal_logprob(u, nvals / 2, math.log(nvals / 8)).sum([1, 2, 3]).view(-1, 1)
            return torch.cat([x, u / nvals], dim=1), logpu
        else:
            raise ValueError()
    else:
        return x, torch.zeros(x.shape[0], 1).to(x)

def remove_padding(x):
    if args.padding > 0:
        return x[:, :im_dim, :, :]
    else:
        return x

# Dataset and hyperparameters
if args.data == 'cifar10':
    im_dim = 3
    n_classes = 10
    
elif args.data == 'mnist':
    im_dim = 1
    init_layer = layers.LogitTransform(1e-6)
    n_classes = 10
    
elif args.data == 'svhn':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    n_classes = 10
    
elif args.data == 'celebahq':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 256:
        args.imagesize = 256
    
elif args.data == 'celeba_5bit':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        args.imagesize = 64
    
elif args.data == 'imagenet32':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 32:
        args.imagesize = 32
    
elif args.data == 'imagenet64':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        args.imagesize = 64
    
n_classes = 1
input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)
