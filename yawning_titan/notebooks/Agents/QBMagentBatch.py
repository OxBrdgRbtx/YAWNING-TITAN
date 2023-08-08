import numpy as np
from yawning_titan.notebooks.Agents.QBMagent import QBMAgent
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv

# import DWAVE model and sampler
from dimod import BinaryQuadraticModel
from dimod import SampleSet
import os
# try:
#     from dwave.system import LeapHybridSampler
# except:
#     pass
from dwave.samplers import SimulatedAnnealingSampler

class QBMBatchAgent(QBMAgent):
#   QBMAgent - An agent that implements a Boltzmann machine (either Reduced BM or Deep BM) based reinforcement learning.

    def __init__(self,*args,**kwargs): # Sample Augmentation options
        super().__init__(*args,**kwargs)

    def setBatchSize(self,batchSize:int=0,maxBits:int=0):
        if batchSize>0:
            self.batchSize = batchSize
        elif maxBits>0:
            self.batchSize = int(maxBits/(self.nHidden*2))
        if maxBits>0 and batchSize>0:
            if self.nHidden * self.batchSize * 2 * self.nParallelAnneals > maxBits:
                print(f'Warning: batchSize * (number of hidden units * 2) * (number of parallel anneals)) ({self.nHidden * self.batchSize * 2 * self.nParallelAnneals}) is larger than the maximum number of bits ({maxBits})')
        
        self.batchRewards = np.zeros((self.batchSize,))
        self.batchStates  = np.zeros((self.batchSize,self.nObservations))
        self.batchStates2  = np.zeros((self.batchSize,self.nObservations))
        self.batchActions = np.zeros((self.batchSize,self.nActions))
        self.batchActions2 = np.zeros((self.batchSize,self.nActions))

    def updateWeights(self,reward,Q1,Q2,state,action,h,state_,action_):
        # Get step modulo batch size and store states/actions
        modStep = (self.step-1) % self.batchSize
        self.batchRewards[modStep]   = reward
        self.batchStates[modStep,:]  = state
        self.batchStates2[modStep,:] = state_
        self.batchActions[modStep,:] = action
        self.batchActions2[modStep,:]= action_

        # If end of batch, then evaluate all QBMs simultaneously
        if modStep == (self.batchSize-1):
            Q1array, harray, Q2array = self.evaluateQBMbatch()
            # Loop through results and update weights
            for iQ in range(self.batchSize):
                super().updateWeights(self.batchRewards[iQ],
                    Q1array[iQ],Q2array[iQ],
                    self.batchStates[iQ,:],self.batchActions[iQ,:],
                    harray[iQ],
                    self.batchStates2[iQ,:],self.batchActions2[iQ,:])
                self.updateQval(self.batchStates[iQ,:],np.argmax(self.batchActions[iQ,:]),Q1array[iQ])
                self.updateQval(self.batchStates2[iQ,:],np.argmax(self.batchActions2[iQ,:]),Q2array[iQ])

    def evaluateQBMbatch(self):
        
        # Initialise Hamiltonian and arrays for individual Hamiltonians
        nQubits = self.nHidden * self.batchSize * 2 * self.nParallelAnneals
        Hamiltonian = BinaryQuadraticModel(nQubits, 'BINARY')

        HamiltonianArray = np.zeros((nQubits,nQubits))
        for iQ in range(self.batchSize):
            # Get Hamiltonians for each batch state/action pair
            thisHamiltonian  = self.buildHamiltonian(self.batchStates[iQ,:] ,self.batchActions[iQ,:])
            thisHamiltonian2 = self.buildHamiltonian(self.batchStates2[iQ,:],self.batchActions2[iQ,:])

            # Stitch individual Hamiltonians together 
            i0 = self.nHidden * self.nParallelAnneals * iQ * 2
            i1 = i0 + self.nHidden * self.nParallelAnneals
            HamiltonianArray[i0:i1,i0:i1] = thisHamiltonian.to_numpy_matrix()

            i0 += self.nHidden * self.nParallelAnneals
            i1 += self.nHidden * self.nParallelAnneals
            HamiltonianArray[i0:i1,i0:i1] = thisHamiltonian2.to_numpy_matrix()

        # Sample batched Hamiltonian
        Hamiltonian = BinaryQuadraticModel.from_numpy_matrix(HamiltonianArray)
        if Hamiltonian.is_linear() and self.explicitRBM:
            pass # Get explicit results - for debug use only
        else:
            results = self.sampleHamiltonian(Hamiltonian,self.SimulateAnneal)
            results = results.to_pandas_dataframe()
        # Process results
        Q1array = np.zeros((self.batchSize,))
        Q2array = np.zeros((self.batchSize,))
        harray  = [np.zeros((self.nHidden,self.nHidden))]*self.batchSize 

        for iQ in range(self.batchSize):
            # Extract relevant hamiltonian for Q1
            i0 = self.nHidden * iQ * 2 * self.nParallelAnneals
            i1 = i0 + self.nHidden * self.nParallelAnneals
            thisHamiltonian= BinaryQuadraticModel.from_numpy_matrix(HamiltonianArray[i0:i1,i0:i1])
            # Get Q, h for this hamiltonian
            if thisHamiltonian.is_linear() and self.explicitRBM:
                Q1array[iQ], harray[iQ] = self.calculateFreeEnergyRBM(self.beta,thisHamiltonian)
            else:
                # Extract relevant sample subsets
                theseSamples = results.iloc[:,i0:i1]
                theseResults = SampleSet.from_samples(theseSamples,'BINARY',thisHamiltonian.energies(theseSamples))
                Q1array[iQ], harray[iQ] = self.calculateFreeEnergy(theseResults,self.beta,thisHamiltonian)
                self.evaluateSampleConvergence(theseResults,self.beta,thisHamiltonian)

            # Extract relevant hamiltonian for Q2
            i0 += self.nHidden * self.nParallelAnneals
            i1 += self.nHidden * self.nParallelAnneals
            thisHamiltonian= BinaryQuadraticModel.from_numpy_matrix(HamiltonianArray[i0:i1,i0:i1])
            # Get Q for this hamiltonian
            if thisHamiltonian.is_linear() and self.explicitRBM:
                Q2array[iQ], _ = self.calculateFreeEnergyRBM(self.beta,thisHamiltonian)
            else:
                # Extract relevant sample subsets
                theseSamples = results.iloc[:,i0:i1]
                theseResults = SampleSet.from_samples(theseSamples,'BINARY',thisHamiltonian.energies(theseSamples))
                Q2array[iQ], _ = self.calculateFreeEnergy(theseResults,self.beta,thisHamiltonian)

        return Q1array, harray, Q2array
