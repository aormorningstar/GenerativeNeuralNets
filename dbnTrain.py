# dbnTrain.py
# source code for training a dbn
# Alan Morningstar
# February 2017

from dbn import dbn
import sys
from dbm import dbm
from utils import neuron
import numpy as np

# inputs
args = sys.argv[1:]
netStr = args[0:-1]
Nstr = netStr[0]
nHstr = netStr[1:]
Tstr = args[-1]
dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + Nstr + '/states_T=' + Tstr + '.csv'

net = [int(n) for n in netStr]
T = float(Tstr)

# output
nHListStr = ''
for n in nHstr:
    nHListStr += '_'+n
nHListStr = nHListStr[1:]

nameBase = '_N=' + Nstr + '_nH=' + nHListStr + '_T=' + Tstr
measurementFileName = 'meas' + nameBase + '.csv'
sampFileName = None

# training parameters
lRPT = 0.002 # learning rate
lR = 0.0002
nEPT = 4000 # number of epochs
nE = 4000
kPT = 10 # k value for CDk or pCD
k = 10
bS = 100 # batch size
method = "CDk" # training method
l1R = 0.0 # L1 regularization
dim = 2 # dimension of Ising model lattice

# sample rbm
numSamples = 10000
numCycles = 1000

# main training function
def main():
    # initialize deep belief network
    model = dbn(net,T,lR,k,bS,nE,l1R,dim)

    # load data
    model.loadData(dataFileName)

    # train dbn
    model.preTrain(lRPT,kPT,nEPT,method)

    # fine tune with wake-sleep
    model.train()

    # generate samples and measure, write to file
    samples = model.sample(numSamples,numCycles)
    model.measure(samples,measurementFileName)

if __name__ == '__main__':
    main()