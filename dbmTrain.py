# dbmTrain.py
# source code for training a deep Boltzmann machine
# Alan Morningstar
# March 2017

from dbm import dbm
import sys
import matplotlib.pyplot as plt
import numpy as np

# inputs
args = sys.argv[1:]
netStr = args[0:-1]
nVstr = netStr[0]
nHstr = netStr[1:]
Tstr = args[-1]
dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + nVstr + '/states_T=' + Tstr + '.csv'
#dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + nVstr + '/states_N='+nVstr+'_T=' + Tstr + '.csv'
#dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_1D_N=' + nVstr + '/states_N=' + nVstr + '_T=' + Tstr + '.csv'

net = [int(n) for n in netStr]
T = float(Tstr)

# output
nHListStr = ''
for n in nHstr:
    nHListStr += '_'+n
nHListStr = nHListStr[1:]

nameBase = '_N=' + nVstr + '_nH=' + nHListStr + '_T=' + Tstr
measurementFileName = 'meas' + nameBase + '.csv'
sampFileName = None

# learning rate
lR = 0.001
lRPT = 0.005
# number of epochs
nE = 6000
nEPT = 2000
# number of Markov chain steps for equilibrating to visibles
nS = 10
# k value for CDk or pCD
k  = 10
kPT = 10
# batch size
bS = 100
# L1 regularization rate
l1R = 0.0
# method
method = 'CDk'
# dimension of lattice
dim = 2

# sampling parameters
nSamples = 10000
nCycles = 1000

# main training function
def main():
    # initialize deep Boltzmann machine
    model = dbm(net,T,lR,nS,k,bS,nE,l1R,dim,roll=True)

    # load data
    model.loadData(dataFileName)

    # pre-train
    model.preTrain(lRPT,kPT,nEPT,method)
    # train dbm
    model.train(method)

    # measurements on dbm samples
    samples = model.sample(nSamples,nCycles,layer=0)
    model.measure(samples,measurementFileName)

if __name__ == '__main__':
    main()
