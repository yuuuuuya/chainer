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


class MLP(Chain):
     def __init__(self, n_units, n_out):
         super(MLP, self).__init__()
         with self.init_scope():
             # the size of the inputs to each layer will be inferred
             self.l1 = L.Linear(None, n_units)  # n_in -> n_units
             self.l2 = L.Linear(None, n_units)  # n_units -> n_units
             self.l3 = L.Linear(None, n_out)    # n_units -> n_out

     def __call__(self, x):
         h1 = F.relu(self.l1(x))
         h2 = F.relu(self.l2(h1))
         y = self.l3(h2)
         return y


def main(args):


    batchSize = args.batch_size
    lamb = args.lamb

    #read datasets in the form of vector.
    train, test = chainer.datasets.get_mnist()

    #extract batSize of data at random
    #repeat=False mean that stop iteration when all example are visited
    train_iter = iterators.SerialIterator(train, batchSize, shuffle=True)
    test_iter = iterators.SerialIterator(test, batchSize, repeat=False, shuffle=False)

    #call Classifer method
    model = L.Classifier(MLP(100, 10))
    #call optimaizer module
    optimizer = optimizers.SGD()
    #setup module prepares for the optimization given a link.
    optimizer.setup(model)
    #regularization
    optimizer.add_hook(chainer.optimizer.WeightDecay(lamb))



    #train
    updater = training.StandardUpdater(train_iter, optimizer)

    trainer = training.Trainer(updater, (20, 'epoch'), out='result')

    #extend module for show corrent train status
    #Evaluates the current model on the test dataset at the end of every epoch
    trainer.extend(extensions.Evaluator(test_iter, model))
    #extensions.LogReport()  accumurates repored values and emits log file
    trainer.extend(extensions.LogReport())
    #print selected items. In this case, 'epoch', 'main/accuracy', 'validation/main/accuracy'
    trainer.extend(extensions.PrintReport(['epoch', 'main/accuracy', 'validation/main/accuracy']))
    #Shows the progress
    trainer.extend(extensions.ProgressBar())
    #run method invoke the training loop
    trainer.run()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'chainer implementation')

    parser.add_argument('--batch_size', '-b', type =int, default =100 )
    parser.add_argument('--lamb', '-l', type =float, default = 0.0005, help = 'regularization coefficient' )
    args = parser.parse_args()

    main(args)
    os.system('say "chainerの実装終了"')
    #os.system('open -a Finder %s'%datapath)
