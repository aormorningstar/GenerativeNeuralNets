# drbn.py
# source code for deep restricted Boltzmann network
# Alan Morningstar
# March 2017

import numpy as np
import pandas as pd
import copy
from dgm import dgm
from dbn import dbn # for pre-training
from utils import neuron
from utils import sigmoid

# a deep restricted Boltzmann network built upon the deep generative model
class drbn(dgm):
    # initialize drbn
    def __init__(self,net,T,lR,k,bS,nE,l1R=0.0,d=2,roll=False):
        # initialize deep generative model framework
        dgm.__init__(self,net,T,lR,k,bS,nE,l1R,d,roll)

        # recognition biases for up pass
        self.bR = copy.deepcopy(self.b)

    # upwards inference
    def upPass(self,state):
        # run upwards over layers in the network
        for i in range(1,self.nL):
            state[i] = neuron( np.dot(state[i-1],self.w[i-1]) + self.bR[i] )

    # downwards inference
    def downPass(self,state):
        # run downwards over layers in the network
        for j in range(self.nL-1):
            i = self.nL-2 - j
            state[i] = neuron( np.dot(state[i+1],self.w[i].transpose()) + self.b[i] )

    # persistent contrastive divergence training step
    def pCD(self):
        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to generate data dependent statistics
        self.upPass(self.dgmState)

        # run persistent chain
        for i in range(self.k):
            self.downPass(self.pC)
            self.upPass(self.pC)

        # update weights
        for i in range(self.nHL):
            self.w[i] += (self.lR/self.bS) * (np.dot(self.dgmState[i].transpose(),self.dgmState[i+1]) - np.dot(self.pC[i].transpose(),self.pC[i+1]))

        # update biases
        for i in range(self.nL):
            self.b[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)
            self.bR[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)

    # contrastive divergence training step
    def CDk(self):
        # clamp visibles to batch of training data
        self.dgmState[0] = self.batch
        # bottom up pass to generate data dependent statistics
        self.upPass(self.dgmState)

        # set chain to drbn state
        self.pC = copy.deepcopy(self.dgmState)
        # run k Gibbs updates
        for i in range(self.k):
            self.downPass(self.pC)
            self.upPass(self.pC)

        # update weights
        for i in range(self.nHL):
            self.w[i] += (self.lR/self.bS) * (np.dot(self.dgmState[i].transpose(),self.dgmState[i+1]) - np.dot(self.pC[i].transpose(),self.pC[i+1]))

        # update biases
        for i in range(self.nL):
            self.b[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)
            self.bR[i] += (self.lR/self.bS) * np.sum(self.dgmState[i]-self.pC[i],axis=0)

    # pre-train using a deep belief network
    def preTrain(self,lR,k,nE,method):
        # initialize deep belief network
        preModel = dbn(self.net,self.T,lR,k,self.bS,nE,self.l1R,self.d,self.roll)
        # load data
        preModel.loadData(self.data)
        # train dbn
        preModel.preTrain(lR,k,nE,method)

        # set pre-trained parameters
        self.w = copy.deepcopy(preModel.w)
        self.b = copy.deepcopy(preModel.b)
        self.bR = copy.deepcopy(preModel.bR)

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
            self.upPass(self.pC)

        # run over epochs
        for e in range(self.nE):
            # print epoch
            if e%10==0:
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
                # regularization
                if self.l1R:
                    self.L1()

                # decrease learning rate
                self.lR -= deltaLR

        # reset learning rate
        self.lR = lRInit

    # samples
    def sample(self,nSamples,nCycles):
        print('----------------------------------')
        print('Sampling dbm...')
        # initialize state of sample dbm
        sampleDgm = [np.random.randint(0,2,(nSamples,self.net[i])) for i in range(self.nL)]

        # run Markov chain to generate equilibrium samples
        for i in range(nCycles):
            self.upPass(sampleDgm)
            self.downPass(sampleDgm)

        # return equilibrium samples
        return sampleDgm[0]

    # use the drbn as a map from the input space of data to a reduced latent space
    def compressedData(self,dataArray,compressedDataFileName = None):
        # data container for compression
        compressDgm = [np.zeros((dataArray.shape[0],self.net[i])) for i in range(self.nL)]
        compressDgm[0] = dataArray

        # propagate probabilities through the network
        for i in range(self.nHL):
            compressDgm[i+1] = sigmoid( np.dot(compressDgm[i],self.w[i]) + self.bR[i+1] )

        # write compressed data
        if compressedDataFileName:
            pd.DataFrame(compressDgm[-1]).to_csv(compressedDataFileName,sep=',',header=False,index=False)

        return compressDgm[-1]
