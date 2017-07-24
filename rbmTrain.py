# rbmTrain.py
# source code for training an rbm
# Alan Morningstar
# July 2017


from rbm import rbm
import sys


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

nameBase = '_N=' + Nstr + '_nH=' + nHListStr + '_T=' + Tstr + '_rbm'
measurementFileName = 'meas' + nameBase + '.csv'
sampFileName = None

# training parameters
lR = 0.001
nE = 100
k = 5
bS = 100 # batch size
method = "CDk" # training method

# sample rbm
numSamples = 10000
numCycles = 1000

# main training function
def main():
    # initialize model
    model = rbm(net,T,lR,k,bS,nE,roll=True)

    # load data
    model.loadData(dataFileName)

    # train model
    model.train()

    # generate samples and measure, write to file
    samples = model.sample(numSamples,numCycles)
    model.measure(samples,measurementFileName)

if __name__ == '__main__':
    main()
