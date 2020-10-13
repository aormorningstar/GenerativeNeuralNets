# dbm.py
# source code for deep Boltzmann machine
# Alan Morningstar
# March 2017


import numpy as np
import pandas as pd
import copy
from dgm import dgm
from dbn import dbn # for pre-training
from utils import neuron


# a deep boltzmann machine built upon the deep generative model
class dbm(dgm):

    # initialize dbm
    def __init__(self,net,T,lR,nS,k,bS,nE,roll=False):

        # initialize deep generative model framework
        dgm.__init__(self,net,T,lR,k,bS,nE,roll)

        # number of Markov steps till convergence of dbm state given data (todo: auto determine this)
        self.nS = nS
        if len(self.net) == 2:
            self.nS = 0


    # pre-train using a deep belief network (not ideal pre-training)
    def preTrain(self,lR,k,nE,method):

        # initialize deep belief network
        preModel = dbn(self.net,self.T,lR,k,self.bS,nE,self.roll)
        # load data
        preModel.loadData(self.data)
        # train dbn
        preModel.preTrain(lR,k,nE,method)

        # set pre-trained parameters
        self.w = copy.deepcopy(preModel.w)
        for i in range(self.nL):
            self.b[i] = 0.5*(copy.deepcopy(preModel.b[i])+copy.deepcopy(preModel.bR[i]))


    # infer odd hidden layers from even layers
    def inferOddHiddens(self,state):

        # run over all odd hidden layers
        for i in range(1,self.nL,2):
            # do differently if top layer
            if i == self.nL-1:
                state[i] = neuron( np.dot(state[i-1],self.w[i-1]) + self.b[i] )
            else:
                state[i] = neuron( np.dot(state[i-1],self.w[i-1]) + np.dot(state[i+1],self.w[i].transpose()) + self.b[i] )


    # infer even hidden layers from odd layers
    def inferEvenHiddens(self,state):

        # run over all even hidden layers
        for i in range(2,self.nL,2):
            # do differently if top layer
            if i == self.nL-1:
                state[i] = neuron( np.dot(state[i-1],self.w[i-1]) + self.b[i] )
            else:
                state[i] = neuron( np.dot(state[i-1],self.w[i-1]) + np.dot(state[i+1],self.w[i].transpose()) + self.b[i] )


    # infer visible layer from 1st hidden layer
    def inferVisibles(self,state):
        state[0] = neuron( np.dot(state[1],self.w[0].transpose()) + self.b[0] )


    # initialize hiddens approximately with a bottom up pass
    def initializeHiddens(self,state):

        # run over hidden layers except top one
        for i in range(1,self.nHL):
            # initialize this layer by doubling the input from the layer below and ignoring layer above
            state[i] = neuron( 2*np.dot(state[i-1],self.w[i-1]) + self.b[i] )

        # initialize top layer
        state[self.nHL] = neuron( np.dot(state[self.nHL-1],self.w[self.nHL-1]) + self.b[self.nHL] )


    # persistent contrastive divergence training step
    def pCD(self):

        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to initialize hidden units approximately
        self.initializeHiddens(self.dgmState)

        # run Markov chain to equilibrate dbm with clamped visibles
        self.inferOddHiddens(self.dgmState)
        for i in range(self.nS):
            self.inferEvenHiddens(self.dgmState)
            self.inferOddHiddens(self.dgmState)

        # run persistent chain to update
        for i in range(self.k):
            self.inferEvenHiddens(self.pC)
            self.inferVisibles(self.pC)
            self.inferOddHiddens(self.pC)

        # update weights
        for i in range(self.nHL):
            self.w[i] += (self.lR/self.bS) * (np.dot(self.dgmState[i].transpose(),self.dgmState[i+1]) - np.dot(self.pC[i].transpose(),self.pC[i+1]))

        # update biases
        for i in range(self.nL):
            self.b[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)


    # contrastive divergence training step
    def CDk(self):

        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to initialize hidden units approximately
        if self.nHL > 1:
            self.initializeHiddens(self.dgmState)

        # run Markov chain to equilibrate dbm with clamped visibles
        self.inferOddHiddens(self.dgmState)
        for i in range(self.nS):
            self.inferEvenHiddens(self.dgmState)
            self.inferOddHiddens(self.dgmState)

        # set persistent chain to dbm state
        self.pC = copy.deepcopy(self.dgmState)
        # run k Gibbs updates
        for i in range(self.k):
            self.inferEvenHiddens(self.pC)
            self.inferVisibles(self.pC)
            self.inferOddHiddens(self.pC)

        # update weights
        for i in range(self.nHL):
            self.w[i] += (self.lR/self.bS) * (np.dot(self.dgmState[i].transpose(),self.dgmState[i+1]) - np.dot(self.pC[i].transpose(),self.pC[i+1]))

        # update biases
        for i in range(self.nL):
            self.b[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)


    # train all parameters together
    def train(self,method):
        # method: 'pCD' or 'CDk'

        # decreasing learning rate
        lRInit = self.lR
        lRFin = self.lR/10.
        deltaLR = (lRInit-lRFin)/(self.nE*self.nB)

        # initialize persistent chain with data
        if method == 'pCD':
            randomIndices = np.random.choice(self.nTS,self.bS,replace=False)
            self.pC[0] = copy.deepcopy(self.data[randomIndices,:])
            self.initializeHiddens(self.pC)

        # run over epochs
        for e in range(self.nE):
            # print epoch
            if e%10 == 0:
                print('----------------------------------')
                print('Training epoch: ',e+1)

            # run over batches
            for b in range(self.nB):
                # set batch to new chunk of data
                self.newBatch()

                # perform a step of contrastive divergence
                if method == 'pCD':
                    self.pCD()
                elif method == 'CDk':
                    self.CDk()

                # decrease learning rate
                self.lR -= deltaLR

        # reset learning rate
        self.lR = lRInit


    # generate samples
    def sample(self,nSamples,nCycles):

        print('----------------------------------')
        print('Sampling dbm...')
        # initialize state of sample dbm
        sampleDgm = [np.random.randint(0,2,(nSamples,self.net[i])) for i in range(self.nL)]

        # run Markov chain to generate equilibrium samples
        for i in range(nCycles):
            self.inferOddHiddens(sampleDgm)
            self.inferEvenHiddens(sampleDgm)
            self.inferVisibles(sampleDgm)

        # return equilibrium samples
        return sampleDgm[0]


    # compress data
    def compressedData(self):

        print('----------------------------------')
        print('Compressing data...')

        # data container for all layers of the network
        state = [np.empty((self.nTS,self.net[i]),dtype=int) for i in range(self.nL)]

        # clamp to data
        state[0] = self.data

        # approx initialize hidden layers
        if self.nHL > 1:
            self.initializeHiddens(state)

        # equilibrate hidden layers
        nCycles = 10*self.nS
        self.inferOddHiddens(state)
        for i in range(nCycles):
            self.inferEvenHiddens(state)
            self.inferOddHiddens(state)

        # return top hidden state equilibrated to data
        return state[-1]
