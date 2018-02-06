import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

import pdb
import argparse
import os.path
import sys



def main(args):


    batchSize = args.batch_size

    #read datasets in the form of vector.
    train, test = chainer.datasets.get_mnist()

    #extract batSize of data at random
    #repeat=False mean that stop iteration when all example are visited
    train_iter = iterators.SerialIterator(train, batchSize, repeat=False, shuffle=True)
    test_iter = iterators.SerialIterator(test, batchSize, repeat=False, shuffle=False)

    pdb.set_trace()







if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'chainer implementation')

    parser.add_argument('--batch_size', '-b', type =int, default =100 )
    parser.add_argument('--figname', '-fig', type =str, default = 'cross' )
    parser.add_argument('--iteration', '-i', type =int, default =200 )
    parser.add_argument('--stepsize', '-s', type =float, default = 0.0000007 )

    args = parser.parse_args()

    main(args)
    os.system('say "chainerの実装終了"')
    #os.system('open -a Finder %s'%datapath)
