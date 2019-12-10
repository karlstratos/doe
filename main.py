# Author: Karl Stratos (me@karlstratos.com)
import argparse
import copy
import inspect
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import util

from collections import OrderedDict


def control_weights(args, models):

    def init_weights(m):
        if hasattr(m, 'weight') and hasattr(m.weight, 'uniform_') and \
           args.init > 0.0:
            torch.nn.init.uniform_(m.weight, a=-args.init, b=args.init)

    for name in models:
        models[name].apply(init_weights)

    # Exactly match estimators for didactice purposes.
    if args.carry == 0:  # MINE(0) == DV
        models['mine'].fXY = copy.deepcopy(models['dv'].fXY)

    if args.alpha == 1:  # INTERPOL(1, *) == CPC
        models['interpol'].fXY = copy.deepcopy(models['cpc'].fXY)
        models['cpc'].transpose = True

    if args.alpha == 0 and args.a == 'e':  # INTERPOL(0, e) == NWJ
        models['interpol'].fXY = copy.deepcopy(models['nwj'].fXY)


def main(args):
    print(args)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if args.cuda else 'cpu')
    pXY = util.CorrelatedStandardNormals(args.dim, args.rho)

    models = {
        'dv': util.SingleSampleEstimator(args.dim, args.hidden, args.layers,
                                         'dv'),
        'mine': util.MINE(args.dim, args.hidden, args.layers,
                          carry_rate=args.carry),
        'nwj': util.SingleSampleEstimator(args.dim, args.hidden, args.layers,
                                          'nwj'),
        'nwjjs': util.NWJJS(args.dim, args.hidden, args.layers),
        'cpc': util.CPC(args.dim, args.hidden, args.layers),
        'interpol': util.Interpolated(args.dim, args.hidden, args.layers,
                                      args.a, args.alpha),
        'doe': util.DoE(args.dim, args.hidden, args.layers, 'gauss'),
        'doe_l': util.DoE(args.dim, args.hidden, args.layers, 'logistic')
    }
    for name in models:
        models[name] = models[name].to(device)
    control_weights(args, models)

    optims = {name: torch.optim.Adam(models[name].parameters(), lr=args.lr)
              for name in models}
    train_MIs = {name: [] for name in models}

    for step in range(1, args.steps + 1):
        X, Y = pXY.draw_samples(args.N)
        X = X.to(device)
        Y = Y.to(device)
        XY_package = torch.cat([X.repeat_interleave(X.size(0), 0),
                                Y.repeat(Y.size(0), 1)], dim=1)
        L = {}
        for name in models:
            optims[name].zero_grad()

            L[name] = models[name](X, Y, XY_package)
            L[name].backward()
            train_MIs[name].append(-L[name].item())

            nn.utils.clip_grad_norm_(models[name].parameters(), args.clip)
            optims[name].step()

        print('step {:4d} | '.format(step), end='')
        for name in L:
            print('{:s}: {:6.2f} | '.format(name, -L[name]), end='')
        print('ln N: {:.2f} | I(X,Y): {:.2f}'.format(math.log(args.N), pXY.I()))

    # Final evaluation
    M = 10 * args.N
    X, Y = pXY.draw_samples(M)
    X = X.to(device)
    Y = Y.to(device)
    XY_package = torch.cat([X.repeat_interleave(M, 0), Y.repeat(M, 1)], dim=1)
    test_MI = {}
    for name in models:
        models[name].eval()
        test_MI[name] = -models[name](X, Y, XY_package)

    print('-'*150)
    print('Estimates on {:d} samples | '.format(M), end='')
    for name in test_MI:
        print('{:s}: {:6.2f} | '.format(name, test_MI[name]), end='')
    print('ln({:d}): {:.2f} | I(X,Y): {:.2f}'.format(M, math.log(M), pXY.I()))

    return test_MI, train_MIs, pXY.I()


def meta_main(args):

    hypers = OrderedDict({
        'hidden': [64, 128, 256],
        'layers': [1],
        'lr': [0.01, 0.003, 0.001, 0.0003],
        'init': [0.0, 0.1, 0.05],
        'clip': [1, 5, 10],
        'carry': [0.99, 0.9, 0.5],
        'alpha': [0.01, 0.5, 0.99],
        'a': ['e'],
        'seed': list(range(100000))
    })

    best_test_MI = {}
    best_train_MIs = {}
    bestargs = {}
    for run_number in range(1, args.nruns + 1):
        if args.nruns > 1:
            print('RUN NUMBER: %d' % (run_number))
            for hyp, choices in hypers.items():
                choice = choices[torch.randint(len(choices), (1,)).item()]
                assert hasattr(args, hyp)
                args.__dict__[hyp] = choice

        test_MI, train_MIs, mi = main(args)
        for name in test_MI:
            if run_number == 1:
                best_test_MI[name] = test_MI[name]
                best_train_MIs[name] = train_MIs[name]
                bestargs[name] = copy.deepcopy(args)
            else:
                if abs(mi - test_MI[name]) < abs(mi - best_test_MI[name]):
                    best_test_MI[name] = test_MI[name]
                    best_train_MIs[name] = train_MIs[name]
                    bestargs[name] = copy.deepcopy(args)

    plt.figure(figsize=(12,5))
    x = range(1, args.steps + 1)
    plt.plot(x, best_train_MIs['doe'], "-r", label='DoE (Gauss)',
             linewidth=0.5)
    plt.plot(x, best_train_MIs['doe_l'], color='tab:orange',
             label='DoE (Logistic)', linewidth=0.5)
    plt.plot(x, best_train_MIs['mine'], "-g", label='MINE', linewidth=0.5)
    plt.plot(x, best_train_MIs['cpc'], "-b", label='CPC', linewidth=0.5)
    plt.plot(x, best_train_MIs['nwj'], "-y", label='NWJ', linewidth=0.5)
    plt.plot(x, best_train_MIs['nwjjs'], "-m", label='NWJ (JS)', linewidth=0.5)
    plt.plot(x, best_train_MIs['interpol'], "-c", label='CPC+NWJ',
             linewidth=0.5)
    plt.plot(x, [mi for _ in range(args.steps)], '-k', label='I(X,Y)')
    plt.plot(x, [math.log(args.N) for _ in range(args.steps)],
             linestyle='dashed', color='0.5', label='ln N')
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylim(-1.5, max(max(best_train_MIs['doe']), mi) + 5)
    plt.xlim(1, args.steps)
    plt.savefig('best_train.pdf', bbox_inches='tight')

    print('-'*150)
    print('Best test estimates on {:d} samples'.format(10 * args.N))
    print('-'*150)
    for name in best_test_MI:
        print('{:10s}: {:6.2f} \t\t {:s}'.format(name, best_test_MI[name],
                                                 str(bestargs[name])))
    print('-'*150)
    print('ln({:d}): {:.2f}'.format(10 * args.N, math.log(10 * args.N)))
    print('I(X,Y): {:.2f}'.format(mi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--N', type=int, default=64,
                        help='number of samples [%(default)d]')
    parser.add_argument('--rho', type=float, default=0.5,
                        help='correlation coefficient [%(default)g]')
    parser.add_argument('--dim', type=int, default=20,
                        help='number of dimensions [%(default)d]')
    parser.add_argument('--hidden', type=int, default=100,
                        help='dimension of hidden states [%(default)d]')
    parser.add_argument('--layers', type=int, default=1,
                        help='number of hidden layers [%(default)d]')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate [%(default)g]')
    parser.add_argument('--init', type=float, default=0.0,
                        help='param init (default if 0) [%(default)g]')
    parser.add_argument('--clip', type=float, default=1,
                        help='gradient clipping [%(default)g]')
    parser.add_argument('--steps', type=int, default=1000, metavar='T',
                        help='number of training steps [%(default)d]')
    parser.add_argument('--carry', type=float, default=0.99,
                        help='EMA carry rate [%(default)g]')
    parser.add_argument('--alpha', type=float, default=1.0,
                        help='interpolation weight (on CPC term) [%(default)g]')
    parser.add_argument('--a', type=str, default='e', choices=['e', 'ff', 'lp'],
                        help='score function in TUBA, INTERPOL [%(default)s]')
    parser.add_argument('--nruns', type=int, default=1,
                        help='number of random runs (not random if set to 1) '
                        '[%(default)d]')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed [%(default)d]')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA?')

    args = parser.parse_args()
    meta_main(args)
