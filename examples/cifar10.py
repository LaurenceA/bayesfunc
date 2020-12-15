import os
import math
import torch as t
import torch.nn as nn
t.set_num_threads(1)
t.backends.cudnn.benchmark=True
t.backends.cudnn.deterministic=False
import torch.nn.functional as F
import argparse
import pandas as pd
from torch.distributions import Categorical
from torchvision import datasets, transforms
from timeit import default_timer as timer

import bayesfunc
from models.resnet8 import net as resnet8

parser = argparse.ArgumentParser()
parser.add_argument('output_filename',     type=str,   help='output filename', nargs='?', default='test')
parser.add_argument('--dataset',           type=str,   help='dataset', nargs='?', default='cifar10')
parser.add_argument('--method',            type=str,   help='method', nargs='?', default='gi')
parser.add_argument('--lr',                type=float, help='learning rate', nargs='?', default=1E-2)
parser.add_argument('--seed',              type=int,   help='seed', nargs='?', default=0)
parser.add_argument('--L',                 type=float, help='temperature scaling', nargs='?', default=1.)
parser.add_argument('--temperL',           action='store_true',  help='temper beta', default=False)
parser.add_argument('--test_samples',      type=int,   help='samples of the weights', nargs='?', default=10)
parser.add_argument('--train_samples',     type=int,   help='samples of the weights', nargs='?', default=1)
parser.add_argument('--prior',             type=str,   help='Prior', nargs='?', default="SpatialIWPrior")
parser.add_argument('--device',            type=str,   help='Device', nargs='?', default="cuda")
parser.add_argument('--batch',             type=int,   help='Batch size', nargs='?', default=500)
parser.add_argument('--subset',            type=int,   help='subset of data size', nargs='?')
parser.add_argument('--epochs',            type=int,   nargs='?', default=1000)
parser.add_argument('--aug', type=str, nargs='?', help='data augmentation, aug or noaug', default="noaug")
args = parser.parse_args()

device = args.device
t.manual_seed(args.seed)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
augment = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

transform_train = {
    'aug'   : transforms.Compose([augment, transform]),
    'noaug' : transform
}[args.aug]

train_dataset = datasets.CIFAR10('data', train=True, download=True, transform=transform_train)
num_classes = max(train_dataset.targets)+1
test_dataset  = datasets.CIFAR10('data', train=False, transform=transform)

train_loader = t.utils.data.DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
test_loader = t.utils.data.DataLoader(test_dataset, batch_size=args.batch)
in_shape = next(iter(train_loader))[0].shape[-3:]

inducing_data, inducing_targets = next(iter(train_loader))
inducing_targets = (t.arange(num_classes) == inducing_targets[:, None]).float() 
inducing_batch = 500

kwargs = {
    'prior'             : getattr(bayesfunc.priors, args.prior),
}
kwargs_lower = {
    'fac' : dict(kwargs),
    'gi'  : dict(kwargs, log_prec_lr=3., inducing_batch=500),
}[args.method]
kwargs_top = {
    'fac' : dict(kwargs),
    'gi'  : dict(kwargs, log_prec_lr=3., log_prec_init=0., inducing_targets=inducing_targets, inducing_batch=500),
}[args.method]


(linear, conv2d) = {
    'gi' : (bayesfunc.GILinear, bayesfunc.GIConv2d),
    'fac' : (bayesfunc.FactorisedLinear, bayesfunc.FactorisedConv2d),
}[args.method]
net = resnet8(linear, conv2d, in_shape, num_classes, kwargs_lower, kwargs_top)

if (args.method == 'gi'):
    net = nn.Sequential(
        bayesfunc.InducingAdd(inducing_batch, inducing_data=inducing_data), 
        net, 
        bayesfunc.InducingRemove(inducing_batch)
    )
net = net.to(device=device)


#initialize with a forward pass
net(next(iter(train_loader))[0].to(device).unsqueeze(0))
opt = t.optim.Adam(net.parameters(), lr=args.lr)


epoch = []
elbo = []
train_ll = []
train_KL = []
test_ll = []
train_correct = []
test_correct = []


def train(epoch):
    iters = 0
    total_elbo = 0.
    total_ll = 0.
    total_KL = 0.
    total_correct = 0.

    if args.temperL and epoch < 100:
        tempered_beta = 0.1*math.floor((epoch-1)/10.)/args.L
    else:
        tempered_beta = 1/args.L

    beta = 1/args.L

    for data, target in train_loader:
        opt.zero_grad()
        data, target = data.to(device), target.to(device)
        data = data.expand(args.train_samples, *data.shape)
        outputs, logPQw, _ = bayesfunc.propagate(net, data)
        outputs = outputs.squeeze(-1).squeeze(-1)

        dist = Categorical(logits=outputs)
        ll = dist.log_prob(target).mean()
        nloss = ll.mean() + tempered_beta * logPQw.mean()/len(train_dataset)  # tempered ELBO
        elbo = ll.mean() + beta * logPQw.mean() / len(train_dataset)
        (-nloss*len(train_dataset)).backward()
        opt.step()

        output = outputs.log_softmax(-1).logsumexp(0) - math.log(outputs.shape[0])
        pred = output.argmax(dim=-1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).float().mean()

        iters         += 1
        total_elbo    += elbo.item()
        total_ll      += ll.item()
        total_KL      -= (beta*logPQw.mean()/len(train_dataset)).item()
        total_correct += correct.item()

    return (total_elbo/iters, total_ll/iters, total_KL/iters, total_correct/iters)

def test():
    iters = 0
    total_elbo = 0.
    total_ll = 0.
    total_correct = 0.
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.expand(args.test_samples, *data.shape)

        outputs = bayesfunc.propagate(net, data)[0].squeeze(-1).squeeze(-1)

        output = outputs.log_softmax(-1).logsumexp(0) - math.log(outputs.shape[0])

        ll = -F.nll_loss(output, target)

        pred = output.argmax(dim=-1, keepdim=True)
        correct = pred.eq(target.view_as(pred)).float().mean()

        iters         += 1
        total_ll      += ll.item()
        total_correct += correct.item()

    return (total_ll/iters, total_correct/iters)


for _epoch in range(args.epochs):
    start_time = timer()
    _elbo, _train_ll, _train_KL, _train_correct = train(_epoch)

    epoch.append(_epoch)
    elbo.append(_elbo)
    train_ll.append(_train_ll)
    train_KL.append(_train_KL)
    train_correct.append(_train_correct)

    with t.no_grad():
        _test_ll, _test_correct = test()
    test_ll.append(_test_ll)
    test_correct.append(_test_correct)

    time = timer() - start_time
    print(f"{os.path.basename(args.output_filename):<32}, epoch:{_epoch:03d}, time:{time: 3.1f}, elbo:{_elbo:.3f}, KL:{_train_KL:.3f}, ll:{_test_ll:.3f}, train_c:{_train_correct:.3f}, test_c:{_test_correct:.3f}", flush=True)

pd.DataFrame({
    'epoch' : epoch,
    'elbo' : elbo,
    'train_ll' : train_ll,
    'train_KL' : train_KL,
    'test_ll' : test_ll,
    'train_correct' : train_correct,
    'test_correct' : test_correct,
}).to_csv(args.output_filename)
