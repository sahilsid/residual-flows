import argparse
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
import torchvision.datasets as vdsets

from lib.resflow import ACT_FNS, ResidualFlow
import lib.datasets as datasets
import lib.optimizers as optim
import lib.utils as utils
import lib.layers as layers
import lib.layers.base as base_layers
from lib.lr_scheduler import CosineAnnealingWarmRestarts

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    '--data', type=str, default='cifar10', choices=[
        'mnist',
        'cifar10',
        'svhn',
        'celebahq',
        'celeba_5bit',
        'imagenet32',
        'imagenet64',
    ]
)
parser.add_argument('--imagesize', type=int, default=32)
parser.add_argument('--nbits', type=int, default=8)  # Only used for celebahq.
parser.add_argument('--block', type=str, choices=['resblock', 'coupling'], default='resblock')

parser.add_argument('--coeff', type=float, default=0.98)
parser.add_argument('--vnorms', type=str, default='2222')
parser.add_argument('--n-lipschitz-iters', type=int, default=None)
parser.add_argument('--sn-tol', type=float, default=1e-3)
parser.add_argument('--learn-p', type=eval, choices=[True, False], default=False)

parser.add_argument('--n-power-series', type=int, default=None)
parser.add_argument('--factor-out', type=eval, choices=[True, False], default=False)
parser.add_argument('--n-dist', choices=['geometric', 'poisson'], default='poisson')
parser.add_argument('--n-samples', type=int, default=1)
parser.add_argument('--batchsize', type=int, default=1)
parser.add_argument('--n-exact-terms', type=int, default=2)
parser.add_argument('--var-reduc-lr', type=float, default=0)
parser.add_argument('--neumann-grad', type=eval, choices=[True, False], default=True)
parser.add_argument('--mem-eff', type=eval, choices=[True, False], default=True)

parser.add_argument('--act', type=str, choices=ACT_FNS.keys(), default='swish')
parser.add_argument('--idim', type=int, default=512)
parser.add_argument('--nblocks', type=str, default='16-16-16')
parser.add_argument('--squeeze-first', type=eval, default=False, choices=[True, False])
parser.add_argument('--actnorm', type=eval, default=True, choices=[True, False])
parser.add_argument('--fc-actnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--batchnorm', type=eval, default=False, choices=[True, False])
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--fc', type=eval, default=False, choices=[True, False])
parser.add_argument('--kernels', type=str, default='3-1-3')
parser.add_argument('--add-noise', type=eval, choices=[True, False], default=True)
parser.add_argument('--quadratic', type=eval, choices=[True, False], default=False)
parser.add_argument('--fc-end', type=eval, choices=[True, False], default=True)
parser.add_argument('--fc-idim', type=int, default=128)
parser.add_argument('--preact', type=eval, choices=[True, False], default=True)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--first-resblock', type=eval, choices=[True, False], default=True)
parser.add_argument('--cdim', type=int, default=256)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--ema-val', type=eval, choices=[True, False], default=True)
parser.add_argument('--task', type=str, choices=['density', 'classification', 'hybrid'], default='density')
parser.add_argument('--scale-dim', type=eval, choices=[True, False], default=False)
parser.add_argument('--rcrop-pad-mode', type=str, choices=['constant', 'reflect'], default='reflect')
parser.add_argument('--padding-dist', type=str, choices=['uniform', 'gaussian'], default='uniform')
parser.add_argument('--resume', type=str, default=None)
parser.add_argument('--nworkers', type=int, default=4)
parser.add_argument('--print-freq', help='Print progress every so iterations', type=int, default=20)
parser.add_argument('--vis-freq', help='Visualize progress every so iterations', type=int, default=500)
args = parser.parse_args()

# Random seed
if args.seed is None:
    args.seed = np.random.randint(100000)

# logger
logger = utils.get_logger(logpath=os.path.join("./", 'logs'), filepath=os.path.abspath(__file__))
logger.info(args)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

if device.type == 'cuda':
    logger.info('Found {} CUDA devices.'.format(torch.cuda.device_count()))
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info('{} \t Memory: {:.2f}GB'.format(props.name, props.total_memory / (1024**3)))
else:
    logger.info('WARNING: Using device {}'.format(device))

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


def update_lr(optimizer, itr):
    iter_frac = min(float(itr + 1) / max(args.warmup_iters, 1), 1.0)
    lr = args.lr * iter_frac
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


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
        logger.info('Changing image size to 256.')
        args.imagesize = 256
    
elif args.data == 'celeba_5bit':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        logger.info('Changing image size to 64.')
        args.imagesize = 64
    
elif args.data == 'imagenet32':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 32:
        logger.info('Changing image size to 32.')
        args.imagesize = 32
    
elif args.data == 'imagenet64':
    im_dim = 3
    init_layer = layers.LogitTransform(0.05)
    if args.imagesize != 64:
        logger.info('Changing image size to 64.')
        args.imagesize = 64
    
logger.info('Creating model.')
n_classes = 1

input_size = (args.batchsize, im_dim + args.padding, args.imagesize, args.imagesize)

if args.squeeze_first:
    input_size = (input_size[0], input_size[1] * 4, input_size[2] // 2, input_size[3] // 2)
    squeeze_layer = layers.SqueezeLayer(2)

# Model
model = ResidualFlow(
    input_size,
    n_blocks=list(map(int, args.nblocks.split('-'))),
    intermediate_dim=args.idim,
    factor_out=args.factor_out,
    quadratic=args.quadratic,
    init_layer=init_layer,
    actnorm=args.actnorm,
    fc_actnorm=args.fc_actnorm,
    batchnorm=args.batchnorm,
    dropout=args.dropout,
    fc=args.fc,
    coeff=args.coeff,
    vnorms=args.vnorms,
    n_lipschitz_iters=args.n_lipschitz_iters,
    sn_atol=args.sn_tol,
    sn_rtol=args.sn_tol,
    n_power_series=args.n_power_series,
    n_dist=args.n_dist,
    n_samples=args.n_samples,
    kernels=args.kernels,
    activation_fn=args.act,
    fc_end=args.fc_end,
    fc_idim=args.fc_idim,
    n_exact_terms=args.n_exact_terms,
    preact=args.preact,
    neumann_grad=args.neumann_grad,
    grad_in_forward=args.mem_eff,
    first_resblock=args.first_resblock,
    learn_p=args.learn_p,
    classification=args.task in ['classification', 'hybrid'],
    classification_hdim=args.cdim,
    n_classes=n_classes,
    block_type=args.block,
)

model.to(device)
ema = utils.ExponentialMovingAverage(model)


def parallelize(model):
    return torch.nn.DataParallel(model)


logger.info(model)
logger.info('EMA: {}'.format(ema))


# Optimization
def tensor_in(t, a):
    for a_ in a:
        if t is a_:
            return True
    return False

if (args.resume is not None):
    logger.info('Resuming model from {}'.format(args.resume))
    with torch.no_grad():
        x = torch.rand(1, *input_size[1:]).to(device)
        model(x)
    checkpt = torch.load(args.resume)
    sd = {k: v for k, v in checkpt['state_dict'].items() if 'last_n_samples' not in k}
    state = model.state_dict()
    state.update(sd)
    model.load_state_dict(state, strict=True)
    ema.set(checkpt['ema'])
    del checkpt
    del state


fixed_z = standard_normal_sample([min(32, args.batchsize),
                                  (im_dim + args.padding) * args.imagesize * args.imagesize]).to(device)



def visualize(model):
    model.eval()
    with torch.no_grad():
        
        # random samples
        fake_imgs = model(fixed_z, inverse=True).view(-1, *input_size[1:])
        if args.squeeze_first: fake_imgs = squeeze_layer.inverse(fake_imgs)
        fake_imgs = remove_padding(fake_imgs)
        fake_imgs = fake_imgs.view(-1, im_dim, args.imagesize, args.imagesize)
        filename   = os.path.join("./", 'samples.png')
        save_image(fake_imgs.cpu().float(), filename, nrow=16, padding=2)
    


def pretty_repr(a):
    return '[[' + ','.join(list(map(lambda i: f'{i:.2f}', a))) + ']]'


def main():
    if args.resume:
        visualize(model)
    

if __name__ == '__main__':
    main()
