# Author: Karl Stratos (me@karlstratos.com)
import argparse
import math
import matplotlib.pyplot as plt
import pickle


def main(args0):

    (args, mi, best_train_MIs, best_test_MI, bestargs) \
        = pickle.load(open(args0.pickle, 'rb'))

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.figure(figsize=(12,5))
    x = range(1, args.steps + 1)
    plt.plot(x, best_train_MIs['doe'], "-r", label='DoE (Gaussian)',
             linewidth=0.5, alpha=1.0)
    plt.plot(x, best_train_MIs['doe_l'], color='tab:orange',
             label='DoE (Logistic)', linewidth=0.5, alpha=1.0)
    plt.plot(x, best_train_MIs['nwj'], "-y", label='NWJ', linewidth=0.3,
             alpha=0.7)
    plt.plot(x, best_train_MIs['nwjjs'], color='tab:brown', label='NWJ (JS)',
             linewidth=0.5, alpha=1.0)
    plt.plot(x, best_train_MIs['dv'], color='tab:pink', label='DV',
             linewidth=0.5)
    plt.plot(x, best_train_MIs['mine'], "-g", label='MINE', linewidth=0.5,
             alpha=1.0)
    plt.plot(x, best_train_MIs['cpc'], "-b", label='CPC', linewidth=0.5,
             alpha=1.0)
    plt.plot(x, best_train_MIs['interpol'], "-c", label='CPC+NWJ',
             linewidth=0.5, alpha=1.0)
    plt.plot(x, [mi for _ in range(args.steps)], '-k', label='I(X,Y)',
             linewidth=0.8)
    plt.plot(x, [math.log(args.N) for _ in range(args.steps)],
             linestyle='dashed', color='0.5', label='ln N',
             linewidth=0.8)
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.ylim(-1.5,
             max(max(best_train_MIs['doe']),
                 max(best_train_MIs['doe_l']),
                 mi) + 5
    )
    plt.xlim(1, args.steps)
    plt.savefig(args0.figure, bbox_inches='tight')

    M = args.c * args.N
    print('-'*150)
    print('Best test estimates on {:d} samples'.format(M))
    print('-'*150)
    for name in best_test_MI:
        print('{:10s}: {:6.2f} \t\t {:s}'.format(name, best_test_MI[name],
                                                 str(bestargs[name])))
    print('-'*150)
    print('ln({:d}): {:.2f}'.format(M, math.log(M)))
    print('I(X,Y): {:.2f}'.format(mi))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('pickle', type=str,
                        help='path to saved pickled file from main.py')
    parser.add_argument('--figure', type=str, default='best_train.pdf',
                        help='output figure file path [%(default)s]')

    args0 = parser.parse_args()
    main(args0)
