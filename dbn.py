# dbn.py
# source code for deep belief network
# Alan Morningstar
# March 2017

import numpy as np
import pandas as pd
from dgm import dgm
from rbm import rbm
from utils import neuron
from utils import sigmoid
import copy

# a deep belief network
class dbn(dgm):
    # initialize dbn
    def __init__(self,net,T,lR,k,bS,nE,l1R=0.0,d=2,roll=False):
        # initialize deep generative model framework
        dgm.__init__(self,net,T,lR,k,bS,nE,l1R,d,roll)

        # recognition weights and biases for compression
        self.wR = copy.deepcopy(self.w)
        self.bR = copy.deepcopy(self.b)

    # layer-wise training of the dbn
    def preTrain(self,lR,k,nE,method):
        # method: 'pCD' or 'CDk'
        print('----------------------------------')
        print('Training...')

        # training data for stack of rbms (layer by layer)
        tData = copy.deepcopy(self.data)
        for i in range(self.nHL):
            print('rbm ',i+1,' of ',self.nHL)
            # initialize stack of rbms (k=1 for all but last rbm)
            RBM = rbm(self.net[i:i+2],self.T,lR,k,self.bS,nE,self.l1R,self.d,roll=False)
            if i == 0 and self.roll:
                RBM.roll = True

            # load training data for this layer
            RBM.loadData(tData)

            # train this layer's rbm
            RBM.train(method)

            # produce training data for next rbm
            tData = RBM.compressedData()

            # update weights of dbn
            self.w[i] = RBM.w[0]
            self.wR[i] = RBM.w[0]
            self.b[i] = RBM.b[0]
            self.bR[i+1] = RBM.b[1]
            if i == 0:
                self.bR[i] = RBM.b[0]
            if i == (self.nHL-1):
                self.b[i+1] = RBM.b[1]

    # fine tune the dbn network with wake-sleep
    def train(self):
        print('----------------------------------')
        print('Fine tuning...')

        # data container for wake phase
        wakeState = [np.random.randint(0,2,(self.bS,self.net[i])) for i in range(self.nL)]
        # data container for sleep phase
        sleepState = [np.random.randint(0,2,(self.bS,self.net[i])) for i in range(self.nL)]

        for e in range(self.nE):
            if (e+1)%10==0:
                print('----------------------------------')
                print("epoch ",e+1," of ",self.nE)

            for b in range(self.nB):
                # set new batch of data
                self.newBatch()
                wakeState[0] = self.batch
                sleepState[0] = self.batch

                ### WAKE PHASE

                # upwards wake update
                self.wakeUpdate(wakeState,sleepState)

                # CDk on top layer RBM
                # data dependent statistics
                wakeState[-1] = neuron( np.dot(wakeState[-2],self.w[-1]) + self.b[-1] )
                # model dependent statistics
                sleepState[-1] = neuron( np.dot(wakeState[-2],self.w[-1]) + self.b[-1] )
                for i in range(self.k):
                    sleepState[-2] = neuron( np.dot(sleepState[-1],self.w[-1].transpose()) + self.b[-2] )
                    sleepState[-1] = neuron( np.dot(sleepState[-2],self.w[-1]) + self.b[-1] )
                # update top layer RBM weights
                self.w[-1] += (self.lR/self.bS) * (np.dot(wakeState[-2].transpose(),wakeState[-1]) - np.dot(sleepState[-2].transpose(),sleepState[-1]))
                # update top layer RBM biases
                self.b[-1] += (self.lR/self.bS) * np.sum(wakeState[-1]-sleepState[-1],axis=0)
                self.b[-2] += (self.lR/self.bS) * np.sum(wakeState[-2]-sleepState[-2],axis=0)

                # downwards sleep update
                sleepState[-2] = neuron( np.dot(sleepState[-1],self.w[-1].transpose()) + self.b[-2] )
                self.sleepUpdate(wakeState,sleepState)

    # wake phase
    def wakeUpdate(self,wakeState,sleepState):
        # run upwards over all but the last layer in the network
        for i in range(1,self.nHL):
            # upards inference
            wakeState[i] = neuron( np.dot(wakeState[i-1],self.wR[i-1]) + self.bR[i] )
            # reconstruction
            sleepState[i-1] = neuron( np.dot(wakeState[i],self.w[i-1].transpose()) + self.b[i-1])

            # update generative weights
            self.w[i-1] += (self.lR/self.bS) * np.dot((wakeState[i-1]-sleepState[i-1]).transpose(),wakeState[i])

            # update generative biases
            self.b[i-1] += (self.lR/self.bS) * np.sum(wakeState[i-1]-sleepState[i-1],axis=0)

    # sleep phase
    def sleepUpdate(self,wakeState,sleepState):
        # run downwards over layers in the network
        for i in range(self.nHL,1,-1):
            # downwards inference
            sleepState[i-1] = neuron( np.dot(sleepState[i],self.w[i-1].transpose()) + self.b[i-1] )
            # reconstruction
            wakeState[i] = neuron( np.dot(sleepState[i-1],self.wR[i-1]) + self.bR[i] )

            # update recognition weights
            self.wR[i-1] += (self.lR/self.bS) * np.dot(sleepState[i-1].transpose(),(sleepState[i]-wakeState[i]))

            # update recognition biases
            self.bR[i] += (self.lR/self.bS) * np.sum(sleepState[i]-wakeState[i],axis=0)

    # generate samples of the dbn
    def sample(self,nSamples,nCycles):
        print('----------------------------------')
        print('Sampling dbn...')
        # initialize state of sample dgm
        sampleDgm = [np.random.randint(0,2,(nSamples,self.net[i])) for i in range(self.nL)]

        # run Markov chain to generate equilibrium samples in top rbm
        for i in range(nCycles):
            sampleDgm[-1] = neuron( np.dot(sampleDgm[-2],self.w[-1]) + self.b[-1] )
            sampleDgm[-2] = neuron( np.dot(sampleDgm[-1],self.w[-1].transpose()) + self.b[-2] )

        # propagate signal down to visibles
        for i in range(self.nHL-2,-1,-1):
            sampleDgm[i] = neuron( np.dot(sampleDgm[i+1],self.w[i].transpose()) + self.b[i] )

        # return equilibrium samples
        return sampleDgm[0]

    # use the dbn as a map from the input space of data to a reduced space at the last layer
    def compressedData(self,dataArray,compressedDataFileName = None):
        # data container for compression
        compressDgm = [np.zeros((dataArray.shape[0],self.net[i])) for i in range(self.nL)]
        compressDgm[0] = dataArray

        # propagate probabilities through the network
        for i in range(self.nHL):
            compressDgm[i+1] = sigmoid( np.dot(compressDgm[i],self.wR[i]) + self.bR[i+1] )

        # write compressed data
        if compressedDataFileName:
            pd.DataFrame(compressDgm[-1]).to_csv(compressedDataFileName,sep=',',header=False,index=False)

        return compressDgm[-1]
