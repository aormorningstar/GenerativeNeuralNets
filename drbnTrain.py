# drbnTrain.py
# source code for training a deep restricted Boltzmann network
# Alan Morningstar
# March 2017

import drbn
import sys

# inputs
args = sys.argv[1:]
netStr = args[0:-1]
nVstr = netStr[0]
nHstr = netStr[1:]
Tstr = args[-1]
dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + nVstr + '/states_T=' + Tstr + '.csv'

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
lR = 0.0002
lRPT = 0.002
# number of epochs
nE = 4000
nEPT = 4000
# k value for CDk or pCD
k  = 10
kPT = 10
# batch size
bS = 100
# L1 regularization rate
l1R = 0.0
# method
method = 'CDk'
methodPT = 'CDk'
# sampling parameters
nSamples = 10000
nCycles = 1000
# dimension of lattice
dim = 2

# main training function
def main():
    # initialize deep Boltzmann machine
    model = drbn.drbn(net,T,lR,k,bS,nE,l1R,dim)

    # load data
    model.loadData(dataFileName)

    # pre-train with dbn
    model.preTrain(lRPT,kPT,nEPT,methodPT)

    # train dbm
    model.train(method)

    # measurements on dbm samples
    sampleVisibles = model.sample(nSamples,nCycles)
    model.measure(sampleVisibles,measurementFileName)

if __name__ == '__main__':
    main()
