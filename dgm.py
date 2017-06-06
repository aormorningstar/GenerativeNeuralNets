# dgm.py
# source code for a deep generative model
# Alan Morningstar
# March 2017

import numpy as np
import pandas as pd
from dataset import dataset
import copy
import pickle

# a deep generative model framework
class dgm(dataset):
    # initialize dgm
    def __init__(self,net,T,lR,k,bS,nE,l1R=0.0,d=2,roll=False):

        # initialize training dataset
        dataset.__init__(self,bS,d,T,roll)

        # network architecture
        self.net = net
        # number of layers in the network
        self.nL = len(self.net)
        # number of hidden layers in the network
        self.nHL = self.nL-1

        # learning rate
        self.lR = lR
        # number of Markov steps for persistent chain each training step
        self.k = k
        # number of training epochs
        self.nE = nE
        # L1 regularization rate
        self.l1R = l1R

        # data container for all layers of the network, initialized randomly
        self.dgmState = [np.random.randint(0,2,(self.bS,self.net[i])) for i in range(self.nL)]
        # persistent chain copy of the network, initialized randomly
        self.pC = [np.random.randint(0,2,(self.bS,self.net[i])) for i in range(self.nL)]

        # weights for the network, initialized to Gaussian with stdDev = 1/sqrt(nV*nH)
        self.w = [(0.1/np.sqrt(self.net[i]*self.net[i+1]))*np.random.randn(self.net[i],self.net[i+1]) for i in range(self.nHL)]
        # biases for the network, initialized to zero
        self.b = [np.zeros((1,self.net[i]), dtype=float) for i in range(self.nL)]

    # generate samples and compute measurements
    def measure(self,samples,measFileName = None):
        # map {0,1} to {-1,1} spin values
        s = 2*samples-1

        # number of visible units, linear size of lattice, and number of samples
        nSamples,nV = s.shape
        L = int(np.sqrt(nV))
        # if a 1D lattice, reset L to it's proper value
        if self.d ==1:
            L = nV

        # average magnetization of samples
        m = np.sum(abs(np.sum(s,1)))/(nSamples*nV)
        # average squared magnetization of samples
        mSquared = np.sum((np.sum(s,1)/nV)**2)/nSamples
        # susceptibility
        chi = nV*(mSquared - m**2.0)/self.T

        # compute energy per spin (ePS) and heat capacity (c)
        ePSList = np.zeros(nSamples)
        # run over individual samples
        for i in range(nSamples):
            # sum up energy contributions from each site
            ePSSum = 0
            if self.d == 2:
                # reshape sample into lattice of spins
                spins = s[i,:].reshape((L,L))
                for j in range(L):
                    for k in range(L):
                        ePSSum -= spins[j,k]*(spins[j-1,k]+spins[j,k-1])
            elif self.d == 1:
                spins = s[i,:]
                for j in range(L):
                    ePSSum -= spins[j]*spins[j-1]
            # add energy per spin of this sample to the list
            ePSList[i] = ePSSum/nV

        # average energy per spin
        ePS = np.sum(ePSList)/nSamples
        # average squared energy per spin
        ePSSquared = np.sum(ePSList*ePSList)/nSamples
        # specific heat
        c = nV*(ePSSquared - ePS**2.0)/(self.T*self.T)

        # print magnetization and susceptability
        print('Temperature    = ',self.T)
        print('Magnetization  = ',m)
        print('Susceptibility = ',chi)
        print('Energy Per Spin = ',ePS)
        print('Specific Heat = ',c)

        # write measurements
        if measFileName:
            meas = np.array([[self.T,ePS,m,c,chi]])
            pd.DataFrame(meas).to_csv(measFileName,header = False, index = False)

    # L1 regularization step
    def L1(self):
        # push weights towards zero
        for i in range(self.nHL):
            self.w[i] -= self.l1R*self.lR*np.sign(self.w[i])

    # save model parameters
    def saveParameters(self,fileName):
        with open(fileName,"wb") as f:
            pickle.dump([self.w,self.b],f)
