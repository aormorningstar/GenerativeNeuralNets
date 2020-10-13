# rbm.py
# some of this code was copied from dbm.py
# Alan Morningstar
# March 2017


import numpy as np
import pandas as pd
import copy
from dgm import dgm
from utils import neuron


# a restricted Boltzmann machine built upon the deep generative model constrained to have one hidden layer
class rbm(dgm):

    # initialize rbm
    def __init__(self,net,T,lR,k,bS,nE,roll=False):

        # initialize deep generative model framework
        dgm.__init__(self,net,T,lR,k,bS,nE,roll)

        # nS is not relevant for a single-hidden-layer network, set to 0
        self.nS = 0
        # rbm network has one hidden layer
        self.nL = 2
        self.nHL = 1


    # infer state of hidden layer
    def inferHiddens(self,state):
        state[1] = neuron( np.dot(state[0],self.w[0]) + self.b[1] )


    # infer state of visible layer
    def inferVisibles(self,state):
        state[0] = neuron( np.dot(state[1],self.w[0].transpose()) + self.b[0] )


    # persistent contrastive divergence training step
    def pCD(self):

        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to initialize hidden units
        self.inferHiddens(self.dgmState)

        # run persistent chain to update
        for i in range(self.k):
            self.inferVisibles(self.pC)
            self.inferHiddens(self.pC)

        # update weights
        self.w[0] += (self.lR/self.bS) * (np.dot(self.dgmState[0].transpose(),self.dgmState[1]) - np.dot(self.pC[0].transpose(),self.pC[1]))

        # update biases
        for i in range(self.nL):
            self.b[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)


    # contrastive divergence training step
    def CDk(self):

        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to initialize hidden units
        self.inferHiddens(self.dgmState)

        # set persistent chain to dbm state
        self.pC = copy.deepcopy(self.dgmState)

        # run k Gibbs updates
        for i in range(self.k):
            self.inferVisibles(self.pC)
            self.inferHiddens(self.pC)

        # update weights
        self.w[0] += (self.lR/self.bS) * (np.dot(self.dgmState[0].transpose(),self.dgmState[1]) - np.dot(self.pC[0].transpose(),self.pC[1]))

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
            self.inferHiddens(self.pC)

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

        # initialize state of sample rbm
        sampleRbm = [np.random.randint(0,2,(nSamples,self.net[i])) for i in range(self.nL)]

        # run Markov chain to generate equilibrium samples
        for i in range(nCycles):
            self.inferHiddens(sampleRbm)
            self.inferVisibles(sampleRbm)

        # return equilibrium samples
        return sampleRbm[0]


    # compress data
    def compressedData(self):

        print('----------------------------------')
        print('Compressing data...')

        # data container for all layers of the network
        state = [np.empty((self.nTS,self.net[i]),dtype=int) for i in range(self.nL)]

        # clamp to data
        state[0] = self.data
        # forward pass to infer hidden data
        self.inferHiddens(state)

        # return hidden state
        return state[1]
