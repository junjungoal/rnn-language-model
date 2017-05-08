import time
import math
import sys
import argparse
import cPickle as pickle
import copy
import os
import codecs
import numpy as np
from chainer import cuda, Variable, FunctionSet, optimizers
import chainer.functions as F

def load_data(args):
    vocab = {}
    print('%s/input.txt'% args.data_dir)
    words = codecs.open('%s/input.txt' % args.data_dir, 'rb', 'utf-8').read()
    words = list(words)
    dataset = np.ndarray((len(words), ), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    print("courpus length: ", len(words))
    print("vocab size: " ,  len(vocab))
    return dataset, words, vocab

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument('--checkpoint_dir',             type=str,   default='cv')
parser.add_argument('--gpu',                        type=int,   default=-1)
parser.add_argument('--rnn_size',                   type=int,   default=128)
parser.add_argument('--learning_rate',              type=float, default=2e-3)
parser.add_argument('--learning_rate_decay',        type=float, default=0.97)
parser.add_argument('--learning_rate_decay_after',  type=int,   default=10)
parser.add_argument('--decay_rate',                 type=float, default=0.95)
parser.add_argument('--dropout',                    type=float, default=0.0)
parser.add_argument('--seq_length',                 type=int,   default=50)
parser.add_argument('--batchsize',                  type=int,   default=50)
parser.add_argument('--epochs',                     type=int,   default=50)
parser.add_argument('--grad_clip',                  type=int,   default=5)
parser.add_argument('--init_from',                  type=str,   default='')

args = parser.parse_args()

n_epochs = args.epochs
n_units = args.rnn_size
batchsize = args.batchsize
bprop_len = args.seq_length
grad_clip = args.grad_clip

train_data, words, vocab = load_data(args)
pickle.dump(vocab, open('%s/vocab.bin'%args.data_dir, 'wb'))

# have to change this line. if there is no args, we will use CharRnn 
model = pickle.load(open(args.init_from, 'rb'))

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Root Mean Square Propagation
# we may replace it with SGD
optimizer = optimizer.RMSprop(lr=args.learning_rate, alpha=args.decay_rate, eps=1e-8)
