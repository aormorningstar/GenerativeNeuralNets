# dataset.py
# source code for a dataset object containing 2D Ising model data
# Alan Morningstar
# March 2017


import numpy as np
import pandas as pd


# a dataset
class dataset(object):

    # initialize dataset
    def __init__(self,bS,T,roll=False):

        # temperature of Ising model data
        self.T = T
        # training data
        self.data = None
        # number of training samples
        self.nTS = 0
        # batch of training data
        self.batch = None
        # batch size
        self.bS = bS
        # track which is the current batch
        self.batchIndex = 0
        # how many batches in the data
        self.nB = 0
        # translationally invariant data?
        self.roll = roll


    # load training data from file or array
    def loadData(self,dataFileOrArray):

        if isinstance(dataFileOrArray,str):
            # load data
            self.data = pd.read_csv(dataFileOrArray,sep=',',header=None).values.astype(int)
        elif isinstance(dataFileOrArray,np.ndarray):
            # set data
            self.data = dataFileOrArray

        # number of training samples and number of spins on the lattice
        self.nTS,self.N = self.data.shape
        # set how many batches are in the data
        self.nB = self.nTS//self.bS


    # shuffle the training data
    def shuffleData(self):

        # shuffle list of data indices
        shuffledIndices = np.random.choice(self.nTS,self.nTS,replace=False)
        self.data = self.data[shuffledIndices,:]


    # shift the training data columns consistently with PBC of lattice, also conjugate spins of half the data
    def rollData(self):

        # roll data columns
        L = int(np.sqrt(self.N))
        r = np.random.randint(L)

        # assume a square lattice
        for i in range(L):
            self.data[:,i*L:(i+1)*L] = np.roll(self.data[:,i*L:(i+1)*L],r,1)

        # conjugate spins of half the data
        conjugatedSamples = np.random.choice(self.nTS,self.nTS//2,replace=False)
        self.data[conjugatedSamples,:] = 1-self.data[conjugatedSamples,:]


    # allocate new data to the batch
    def newBatch(self):

        if self.batchIndex < (self.nB-1):
            # push batch index to next
            self.batchIndex += 1
        else:
            # shuffle data set
            self.shuffleData()
            # roll if necessary
            if self.roll:
                self.rollData()
            # reset index to first batch
            self.batchIndex = 0

        # first row of the new batch
        firstRow = self.bS*self.batchIndex
        # set batch to chunk of data
        self.batch = self.data[firstRow:firstRow + self.bS,:]
