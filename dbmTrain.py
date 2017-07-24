# dbmTrain.py
# source code for training a deep Boltzmann machine
# Alan Morningstar
# March 2017


from dbm import dbm
import sys


# inputs
args = sys.argv[1:]
netStr = args[0:-1]
nVstr = netStr[0]
nHstr = netStr[1:]
Tstr = args[-1]
dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + nVstr + '/states_T=' + Tstr + '.csv'
#dataFileName = '/Users/aormorningstar/Documents/PSI/Essay/data/data_N=' + nVstr + '/states_N='+nVstr+'_T=' + Tstr + '.csv'

net = [int(n) for n in netStr]
T = float(Tstr)

# output
nHListStr = ''
for n in nHstr:
    nHListStr += '_'+n
nHListStr = nHListStr[1:]

nameBase = '_N=' + nVstr + '_nH=' + nHListStr + '_T=' + Tstr + '_dbm'
measurementFileName = 'meas' + nameBase + '.csv'
sampFileName = None

# learning rate
lR = 0.001
lRPT = 0.005
# number of epochs
nE = 20
nEPT = 20
# number of Markov chain steps for equilibrating to visibles
nS = 10
# k value for CDk or pCD
k  = 5
kPT = 5
# batch size
bS = 100
# method
method = 'CDk'

# sampling parameters
nSamples = 10000
nCycles = 1000

# main training function
def main():
    # initialize model
    model = dbm(net,T,lR,nS,k,bS,nE,roll=True)

    # load data
    model.loadData(dataFileName)

    # pre-train
    model.preTrain(lRPT,kPT,nEPT,method)

    # train dbm
    model.train(method)

    # measurements on model samples
    samples = model.sample(nSamples,nCycles)
    model.measure(samples,measurementFileName)

if __name__ == '__main__':
    main()
