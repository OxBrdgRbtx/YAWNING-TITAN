import numpy as np
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.notebooks.Outputs.qbmLogger import qbmLogger
from yawning_titan.notebooks.Outputs.QBMResults import QBMResults

# import DWAVE model and sampler
from dimod import BinaryQuadraticModel
import os
from dwave.system import FixedEmbeddingComposite, LazyFixedEmbeddingComposite, DWaveSampler

from dimod import SimulatedAnnealingSampler
import json

class QBMAgent:
#   QBMAgent - An agent that implements a Boltzmann machine (either Reduced BM or Deep BM) based reinforcement learning.
#   Functions:
#   initRBM             - Initialises a reduced boltzmann machine
#   initDBM             - Initialises a deep boltzmann machine
#   loadWeights         - Initialises a generic BM based on saved weight .txt files. Assumes any zero weights are unconnected nodes
#   buildHamiltonian    - Builds a hamiltonian to represent the free energy in the machine, either state and action, or state only are acceptable inputs
#   calculateFreeEnergy - Calculates the free energy from the hamiltonian and samples of the hamiltonian
#   evaluateQBM         - Evaluates the boltzmann machine - builds hamiltonian and calculates the energy
#   sampleHamiltonian   - Sample Hamiltonian via either simulated Annealing or Quantum Annealing
#   getOptimalAction    - Find the optimal action for a given state. Option to sample BM for lowest energy, or directly choose from saved Q values
#   getLowEnergyAction  - Find the action that has the lowest energy from a sampled hamiltonian (with action nodes free)
#   updateQval          - Update the stored Q value for a state, action pair
#   getStateI           - Get the index of a given state for use in the Q matrix
#   updateWeights       - Update the weights of the BM
#   scaleReward         - Scale the reward (forward or backward), useful to improve BM fitting
#   AmendZeroState      - Amend a state of all zeros to a state of all zeros, with a 1 final node flag
#   ChooseAction        - Choose an action and evaluate the Q value from the BM
#   learn               - Run the training loop

    def __init__(self,env: GenericNetworkEnv,saveName="QBM",
        beta:float=5,SAbeta:float=2,epsilon:float=1e-2,epsilon0:float=1e-5,gamma:float=0.9, # Learning hyperparameters
        nRandomSteps:int=0,pRandomDecay:float=0.9999,minPrandom:float=0.001, # Random action choice parameters
        printRate:int=10000,gameWindow:int=100,stepWindow:int=10000, # Logging options
        writeStepLogs:bool=True,writeGameLogs:bool=True,writeWeights:bool=False, # Logging flags
        writeToTerminal:bool=True,writeTerminalToText:bool=True, # More logging flags
        SimulateAnneal:bool=True,AnnealToBestAction:bool=False,SimulateAnnealForAction=True,  # Annealing flags
        adaptiveGradient:bool=True,explicitRBM:bool=True, # Annealing flags
        AugmentSamples:bool=True,nParallelAnneals:int=1, # Sampling assistance options
        numReads:int=100,convergenceCheckInterval:int=200, # Sample size options
        maximumQPUminutes:float=60,DWave_solver='Advantage_system4.1'): 


        self.env = env
        self.name = saveName

        # Get number of actions/observations
        self.nActions = env.blue_actions
        self.nObservations = env.observation_space.shape[0] + 1
        self.nVisible = self.nActions + self.nObservations

        # Learning hyperparameters
        self.beta = beta # Thermodynamic beta in free energy calculations
        self.SAbeta = SAbeta # Thermodynamic beta in simulated annealing
        self.epsilon = epsilon # Update step length
        self.epsilon0 = epsilon0 # For use with adaptive gradient - minimum epsilon
        self.gamma = gamma # Weight on future reward

        # Random action choice parameters
        self.nRandomSteps = nRandomSteps # Number of random steps to take in initial learning
        self.pRandomDecay = pRandomDecay # Stepwise decay on p(take random step) after nRandomSteps
        self.pRandom = 1 # Initial p(take random step) value
        self.minPrandom = minPrandom # Minimum p(take random step) value
        
        # Logging
        self.log = qbmLogger(saveName=saveName,
            printRate=printRate,
            gameWindow=gameWindow,
            stepWindow=stepWindow,
            writeStepLogs=writeStepLogs, # Write summary step logs to .txt files periodically during training
            writeGameLogs=writeGameLogs, # Write summary game logs to .txt files after each game
            writeWeights=writeWeights, # Save weights to .txt files periodically during training
            writeToTerminal=writeToTerminal, # Print results to terminal periodically
            writeTerminalToText=writeTerminalToText) # Write printed terminal output to text file

        # Internal counters
        self.step = 0
        self.outerGradSum = []
        self.batchSize = None

        # Set flags
        self.QBMinitialised = False # Flag to check nodes are set up
        self.SimulateAnneal = SimulateAnneal # Simulate or use D-Wave for updating Q
        self.SimulateAnnealForAction = SimulateAnnealForAction # Simulate or use D-Wave for choosing action
        self.adaptiveGradient = adaptiveGradient # Flag to use adaptive gradient method
        self.AnnealToBestAction = AnnealToBestAction # Flag to choose best action via annealing rather than reviewing Q values
        self.explicitRBM = explicitRBM # Solve RBM equations explicitly, without sampling
        # Set up samplers
        self.simulatedSampler = SimulatedAnnealingSampler()
        if SimulateAnneal or SimulateAnnealForAction:
            if (not SimulateAnneal) or (not SimulateAnnealForAction):
                # First if case makes this exclusive OR
                self.quantumSampler = [LazyFixedEmbeddingComposite(DWaveSampler(solver=DWave_solver))]
                self.embeddingLoaded = [False]
                self.DWave_solver = DWave_solver
        else:
            # (not SimulateAnneal) AND (not SimulateAnnealForAction)
            # Need two different embeddings
            self.quantumSampler = [LazyFixedEmbeddingComposite(DWaveSampler(solver=DWave_solver)), LazyFixedEmbeddingComposite(DWaveSampler(solver=DWave_solver))]
            self.embeddingLoaded = [False,False]
            self.DWave_solver = DWave_solver
        # Augment sample options
        # For each sampled state, create copies of the state, where each node is switched, and each pair of nodes is switched
        self.AugmentSamples = AugmentSamples
        self.nParallelAnneals = nParallelAnneals # Number of parallel solves of the same Hamiltonian
        
        # Number of reads and convergence check options
        self.numReads = numReads # number of QPU reads
        self.convergenceCheckInterval = convergenceCheckInterval # number of steps between convergence checks
        self.lastConvergenceCheck = 0

        # QPU limits
        self.maximumQPUminutes = maximumQPUminutes # maximum QPU time to spend

        # Set Stored Q maximums
        self.storeQ = self.nObservations<26
        if not self.AnnealToBestAction and not self.storeQ:
            print('Warning: too many observables to store Q values. ''AnnealToBestAction'' flag has been set to ''True''')
            self.AnnealToBestAction = True
        if self.storeQ:
            self.Qvals = np.ones(shape=(2**(self.nObservations-1),self.nActions))*np.nan
            self.stateMult = [2**i for i in range(self.nObservations)]
  
    def initRBM(self,hiddenNodes = 5):
        # Initialise an RBM with random weights, zero weight between hidden nodes
        self.nHidden = hiddenNodes
        self.hiddenLayerSizes = [hiddenNodes]
        self.hvWeights = np.random.normal(loc=0.0,scale=1e-3,size=(self.nVisible,self.nHidden))
        self.hhWeights = np.zeros((self.nHidden,self.nHidden))

        self.nonzeroHV = np.ones_like(self.hvWeights)
        self.nonzeroHH = np.zeros_like(self.hhWeights)

        self.QBMinitialised = True

    def initDBM(self,hiddenNodes = [5,5]):
        # Initialise an DBM with random weights
        self.nHidden = sum(hiddenNodes)
        self.hiddenLayerSizes = hiddenNodes
        self.hvWeights = np.random.normal(loc=0.0,scale=1,size=(self.nVisible,self.nHidden))
        self.hhWeights = np.random.normal(loc=0.0,scale=1,size=(self.nHidden,self.nHidden))

        self.nonzeroHV = np.zeros_like(self.hvWeights)
        self.nonzeroHV[0:self.nObservations,0:hiddenNodes[0]] = 1
        self.nonzeroHV[self.nObservations:self.nVisible,-hiddenNodes[-1]:] = 1
        self.hvWeights = np.multiply(self.hvWeights,self.nonzeroHV)

        # Strictly upper triangular matrix to avoid duplication
        self.nonzeroHH = np.zeros_like(self.hhWeights)
        cumsum0 = 0
        for iH, nLayer in enumerate(hiddenNodes[:-1]):
            cumsum1 = cumsum0 + nLayer
            self.nonzeroHH[cumsum0:cumsum1,cumsum1:(cumsum1+hiddenNodes[iH+1])] = 1
            cumsum0 = cumsum1
        self.hhWeights = np.multiply(self.hhWeights,self.nonzeroHH)

        # Scale weights so initial free energy is between -1 and 1
        scale = np.sum(np.abs(self.hhWeights)) + np.sum(np.abs(self.hvWeights))
        self.hvWeights = self.hvWeights / scale
        self.hhWeights = self.hhWeights / scale

        self.QBMinitialised = True

    def loadWeights(self,resultsDir):
        self.hvWeights = np.loadtxt(os.path.join(resultsDir,'hvWeights.txt'),delimiter=',')
        self.hhWeights = np.loadtxt(os.path.join(resultsDir,'hhWeights.txt'),delimiter=',')

        self.nHidden = np.shape(self.hvWeights)[0]
        self.nonzeroHV = (self.hvWeights != 0).astype(int)
        self.nonzeroHH = (self.hhWeights != 0).astype(int)
        self.QBMinitialised = True

        self.outerGradSum[0] = np.loadtxt(os.path.join(resultsDir,'hvGradscale.txt'),delimiter=',')
        self.outerGradSum[1] = np.loadtxt(os.path.join(resultsDir,'hhGradscale.txt'),delimiter=',')



    def saveWeights(self,resultsDir):
        np.savetxt(os.path.join(resultsDir,'hvWeights.txt'),self.hvWeights,delimiter=',')
        np.savetxt(os.path.join(resultsDir,'hhWeights.txt'),self.hhWeights,delimiter=',')
        np.savetxt(os.path.join(resultsDir,'hvGradscale.txt'),self.outerGradSum[0],delimiter=',')
        np.savetxt(os.path.join(resultsDir,'hhGradscale.txt'),self.outerGradSum[1],delimiter=',')

    def buildHamiltonian(self,state,action):
        if action == []:
            # No action recieved, include in Hamiltonian
            freeAction = True
            nFreeNodes = self.nHidden + self.nActions
            observation = state
        else:
            # Action and state received, build hamiltonian on hidden nodes only and add bias
            freeAction = False
            nFreeNodes = self.nHidden
            observation = np.concatenate((state,action))
        Hamiltonian = BinaryQuadraticModel(nFreeNodes, 'BINARY')
        # Build Hamiltonian
        quadTerms = np.zeros((nFreeNodes,nFreeNodes))
        quadTerms[:self.nHidden,:self.nHidden] = -self.hhWeights
        if not freeAction:
            linearTerms = -np.matmul(np.transpose(self.hvWeights),observation)
        else:
            linearTerms = np.zeros((nFreeNodes,))
            linearTerms[:self.nHidden] = -np.matmul(np.transpose(self.hvWeights[:self.nObservations,:]),observation)
            quadTerms[self.nHidden:,:self.nHidden] = -self.hvWeights[self.nObservations:,:]
            
            # Add penalty term to ensure one node is activated
            penaltyScale = 20
            linearTerms[self.nHidden:] = -penaltyScale
            quadTerms[self.nHidden:,self.nHidden:] = np.triu(2 * penaltyScale * np.ones((self.nActions,self.nActions)),1)

        # Create block off-diagonal quadratic array
        quadTermsBlock = np.zeros((nFreeNodes * self.nParallelAnneals,nFreeNodes * self.nParallelAnneals))
        i0 = 0
        for iP in range(self.nParallelAnneals):
            i1 = i0 + nFreeNodes
            quadTermsBlock[i0:i1,i0:i1] = quadTerms
            i0 += nFreeNodes
        Hamiltonian.add_linear_from_array(np.tile(linearTerms,self.nParallelAnneals))
        Hamiltonian.add_quadratic_from_dense(quadTermsBlock)

        return Hamiltonian

    def calculateFreeEnergy(self,results,beta,Hamiltonian):
        # Take results from sampled hamiltonian and calculate mean free energy

        # Drop copies of the hamiltonian
        if len(Hamiltonian)>self.nHidden:
            for v in range(self.nHidden,self.nHidden*self.nParallelAnneals):
                Hamiltonian.remove_variable(v)

        # Add additional samples near the minima to help in calculating mean
        if self.AugmentSamples:
            # Collate results from copies of the hamiltonian into equvalent results for the baseline hamiltonian
            if type(results) is list:
                records = [[results[x*self.nHidden:(x+1)*self.nHidden],
                            Hamiltonian.energy(results[x*self.nHidden:(x+1)*self.nHidden])] for x in range(self.nParallelAnneals)]
            else:
                resAgg0 = results.aggregate()
                records = resAgg0.record
                records = [[record[0][x*self.nHidden:(x+1)*self.nHidden],
                            Hamiltonian.energy(record[0][x*self.nHidden:(x+1)*self.nHidden])] for x in range(self.nParallelAnneals) for record in records]
            resAug = []
            sampAgg = []
            for record in records:
                resAug += [[record[0],record[1]]]
                sampAgg += [record[0]]

                # First augment by flipping each bit individually
                for iFlip in range(self.nHidden):
                    flipBit = np.zeros(self.nHidden)
                    flipBit[iFlip] = 1
                    thisSamp = np.mod(record[0] + flipBit,2)
                    thisEnergy = Hamiltonian.energy(thisSamp)
                    resAug += [[thisSamp,thisEnergy]]
                    sampAgg += [thisSamp]

                    # # Then flip pairs of bits
                    # for iFlip2 in range(iFlip+1,self.nHidden):
                    #     flipBit = np.zeros(self.nHidden)
                    #     flipBit[iFlip] = 1
                    #     flipBit[iFlip2] = 1
                    #     thisSamp = np.mod(record[0] + flipBit,2)
                    #     thisEnergy = Hamiltonian.energy(thisSamp)
                    #     resAug += [[thisSamp,thisEnergy]]
                    #     sampAgg += [thisSamp]
                
            _,uInds = np.unique(np.array(sampAgg),axis=0,return_index=True)
            resAgg = [resAug[iU] for iU in uInds]
        else:
            if type(results) is list:
                records = [[results[x*self.nHidden:(x+1)*self.nHidden],
                            Hamiltonian.energy(results[x*self.nHidden:(x+1)*self.nHidden])] for x in range(self.nParallelAnneals)]
                resAgg = [[results,Hamiltonian.energy(results)]]
            else:
                resAgg0 = results.aggregate()
                records = resAgg0.record
                records = [[record[0][x*self.nHidden:(x+1)*self.nHidden],
                            Hamiltonian.energy(record[0][x*self.nHidden:(x+1)*self.nHidden])] for x in range(self.nParallelAnneals) for record in records]
                resAgg = results.aggregate().record
            states = [record[0] for record in records]
            _,uInds = np.unique(states,axis=0,return_index=True)
            resAgg = [records[iU] for iU in uInds]

        # Calculate Zv
        HamAvg = 0.0
        entropy = 0.0
        Zv = 0.0
        h = np.zeros((self.nHidden,self.nHidden))
        energies = [record[1] for record in resAgg]
        minEnergy = min(energies) # Stop overflowing exp

        # Get proportional probability for all samples, sum for scalar
        proportionalProbability = np.exp(-(energies-minEnergy) * beta)
        this_Zv = proportionalProbability.sum()

        # Filter samples with negligible probabilities
        keepSamples = proportionalProbability/this_Zv > (1e-5/len(energies))
        resAgg = [record for record,keep in zip(resAgg,keepSamples) if keep]
        proportionalProbability = [thisP for thisP,keep in zip(proportionalProbability,keepSamples) if keep]
        Zv = np.sum(proportionalProbability)

        # For each record, calculate probability, mean hamiltonian contribution,
        # entropy contribution and h interaction contributions
        
        # theseProbabilities = proportionalProbability / Zv
        # theseEnergies = [record[1] for record in resAgg]

        # HamAvg = np.dot(theseProbabilities,theseEnergies)
        # entropy = np.sum(theseProbabilities * np.log(theseProbabilities))
        # theseSamples = []
        # for iH in range(self.nHidden):
        #     theseSamples += [[record[0][iH] for record in resAgg]]
        # for iH in range(self.nHidden):
        #     h[iH,iH] = np.dot(theseProbabilities,theseSamples[iH])
        #     for iH2 in range(iH+1,self.nHidden):
        #         theseSamples_ = [record[0][iH2] for record in resAgg]
        #         h[iH,iH2] = np.dot(theseProbabilities,np.multiply(theseSamples[iH], theseSamples[iH2]))
                
        for record in resAgg:
            pEnergy = np.exp(-(record[1]-minEnergy) * beta)/Zv
            HamAvg += record[1] * pEnergy
            entropy += pEnergy * np.log(pEnergy)
            h += pEnergy * np.diag(record[0])
            h += pEnergy * np.multiply(np.outer(record[0],record[0]),self.nonzeroHH)

        # for record in resAgg:
        #     pEnergy = np.exp(-(record[1]-minEnergy) * beta)/Zv
        #     if pEnergy>0:
        #         HamAvg += record[1] * pEnergy
        #         entropy += pEnergy * np.log(pEnergy)
        #         for iH in range(self.nHidden):
        #             h[iH,iH] += pEnergy * record[0][iH]
        #             for iH2 in range(iH+1,self.nHidden):
        #                 h[iH,iH2] += pEnergy * record[0][iH] * record[0][iH2]

        minusF = - HamAvg - 1/beta * entropy
        return minusF, h

    def calculateFreeEnergyRBM(self,beta,Hamiltonian):
        linTerms = np.diag(Hamiltonian.to_numpy_matrix())
        pEnergy = (np.exp(linTerms)+1)**-1
        h = np.diag(pEnergy)

        pEnergy_ = pEnergy[np.logical_and(pEnergy!=0,pEnergy!=1)]
        entropy = np.matmul(pEnergy_,np.log(pEnergy_)) + np.matmul(1-pEnergy_,np.log(1-pEnergy_))
        
        minusF = -np.matmul(pEnergy,linTerms) - 1/beta * entropy

        return minusF, h

    def evaluateQBM(self,state,action,SimulateAnneal=True):
        # This function receives the current state and action choice
        # It then calculates the value of -F, and the average h values for use in weight updates

        # Build Hamiltonian
        Hamiltonian = self.buildHamiltonian(state,action)

        # If all hamiltonian terms are linear, then we are solving a clamped RBM, which has an explicit solution
        # Solve explicitly if desired
        if Hamiltonian.is_linear() and self.explicitRBM:
            minusF, h = self.calculateFreeEnergyRBM(self.beta,Hamiltonian)
            return minusF, h

        # Sample Hamiltonian and aggregate results
        results = self.sampleHamiltonian(Hamiltonian,SimulateAnneal)

        if action == []:
            # No action supplied, searching for the best choice of action
            actionI,resultSample = self.getLowEnergyAction(results)
            Hamiltonian = self.buildHamiltonian(state,resultSample[self.nHidden:])
            minusF, h = self.calculateFreeEnergy(resultSample[:self.nHidden],self.beta,Hamiltonian)
            return actionI, minusF, h
        else:
            # Process results and calculate mean energy, h
            minusF, h = self.calculateFreeEnergy(results,self.beta,Hamiltonian)
            self.evaluateSampleConvergence(results,self.beta,Hamiltonian)
            return minusF, h

    def evaluateSampleConvergence(self,results,beta,Hamiltonian):
        if self.step - self.lastConvergenceCheck < self.convergenceCheckInterval:
            return # Skip unless interval has passed
        self.lastConvergenceCheck = self.step


        nSamples = len(results.record)
        # Check for convergence
        checkStep = round(0.05 * nSamples) # checking in blocks of 5 %
        if (checkStep == 0):
            checkStep = 1
        convergenceTolerance = 0.01 # 1 percent tolerance of convergence
        checkRange = int(np.ceil(nSamples/checkStep)) # currently checks all samples - adjust

        energyVals = []
        for i in range(checkRange):
            # compare the average free energy in adjacent blocks of size checkStep
            minusF, h = self.calculateFreeEnergy(results.truncate(np.min([checkStep*(i+1),nSamples])),self.beta,Hamiltonian) # check access of results
            energyVals += [minusF]

        relativeDifference = abs(energyVals-energyVals[-1])/energyVals[-1]
        relativeDifferenceSmall = relativeDifference<convergenceTolerance
        convergedIndex = min(idx for idx, val in enumerate(relativeDifferenceSmall) if val) + 1
        convergedIndexPct = convergedIndex * checkStep / nSamples
        if convergedIndexPct <= 0.4:
            # Converged with less than 40% samples
            self.numReads = max(int(self.numReads * 0.8),50) # Avoid numReads getting too small
        if convergedIndexPct <= 0.6:
            # Converged with 40-60% samples
            self.numReads = max(int(self.numReads * 0.9),50) # Avoid numReads getting too small
        elif convergedIndexPct >= 0.9:
            # Converged with 90-100% samples, or hasn't converged
            self.numReads = min(int(self.numReads * 1.2),500) # Avoid numReads getting too large
        elif convergedIndexPct >= 0.8:
            # Converged with 80-90% samples
            self.numReads = min(int(self.numReads * 1.1),500) # Avoid numReads getting too large
        self.numReads = round(self.numReads)

    def sampleHamiltonian(self,Hamiltonian,SimulateAnneal):
        if SimulateAnneal:
            beta0 = min(0.1,self.SAbeta/5)
            results = self.simulatedSampler.sample(Hamiltonian,num_reads=self.numReads,beta_range=[beta0, self.SAbeta],num_sweeps=20) 
        else:
            nHidden = self.nHidden * self.nParallelAnneals
            if self.batchSize is not None:
                nHidden = self.nHidden * 2 * self.batchSize * self.nParallelAnneals
            if len(Hamiltonian) == nHidden:
                iSampler = 0
            else:
                iSampler = 1 # If sampling for action, need two separate fixed embeddings
            try:
                if not self.embeddingLoaded[iSampler]:
                    if self.batchSize is None:    
                        embedding_name = 'hidden_'+'_'.join([str(x) for x in self.hiddenLayerSizes]) + \
                            '_numParallel_'+str(self.nParallelAnneals)+'_sampler_'+str(iSampler)
                    else:
                        embedding_name = 'hidden_'+'_'.join([str(x) for x in self.hiddenLayerSizes]) + \
                            '_Batch_'+str(self.batchSize)+'_numParallel_'+str(self.nParallelAnneals)+'_sampler_'+str(iSampler)
                    embedding_name = 'embeddings\\'+self.DWave_solver+'\\'+embedding_name+'.txt'
                    if os.path.isfile(embedding_name):
                        self.loadEmbedding(iSampler,embedding_name)
                results = self.quantumSampler[iSampler].sample(Hamiltonian,num_reads=self.numReads)
                if not self.embeddingLoaded[iSampler]:
                    # Doesn't yet exist - save
                    self.saveEmbedding(iSampler,embedding_name)
                    self.embeddingLoaded[iSampler] = True
                self.log.addQPUtime(results.info["timing"]["qpu_access_time"])
            except:
                if not self.embeddingLoaded[iSampler]:
                    print(e_for_error)
                beta0 = min(0.1,self.SAbeta/5)
                results = self.simulatedSampler.sample(Hamiltonian,num_reads=self.numReads,beta_range=[beta0, self.SAbeta],num_sweeps=20)
                self.log.QPUfail(self.step)
        return results

    def getOptimalAction(self,state):
        stateI = self.getStateI(state)

        if self.AnnealToBestAction:
            # Sample Hamiltonian to get the action with the lowest energy (highest reward)
            actionI, Q, h = self.evaluateQBM(state,[],self.SimulateAnnealForAction)
            if self.SimulateAnnealForAction and not self.SimulateAnneal:
                # Chosen action via SA, but want Q from QA
                Q = []
                h = []
        elif any(np.isnan(self.Qvals[stateI])):
            # Haven't evaluated Q for all actions yet
            Q = self.Qvals[stateI]
            hA = [[]]*self.nActions
            # Loop over actions, calculate Q for those without a stored value
            for iA in range(self.nActions):
                if np.isnan(Q[iA]):
                    action = np.zeros(self.nActions)
                    action[iA] = 1
                    Q[iA], hA[iA] = self.evaluateQBM(state,action,self.SimulateAnnealForAction)

            # Store optimal Q index and value
            self.Qvals[stateI][:] = Q
            
            # Choose actionI to maximise Q
            actionI = np.argmax(Q)
            if hA[actionI]==[]:
                # If best Q seen before, don't return Q as QBM needs to be evaluated for latest value
                Q = []
                h = []
            else:
                # If best Q is new, return Q, h to avoid repeating calculations
                Q = Q[actionI]
                h = hA[actionI]
        else:
            # Have seen this state - choose maximum stored Q
            # Don't return Q - need to evaluate the QBM for the latest value
            actionI = np.argmax(self.Qvals[stateI])
            Q = []
            h = []
        return actionI, Q, h

    def getLowEnergyAction(self,results):
        # From a set of results, find the lowest energy state and extract the action performed
        LEsamp = results.first
        LEaction = [iV for iV in range(self.nHidden,self.nHidden+self.nActions) if LEsamp[0][iV]==1]

        if len(LEaction) != 1:
            print(f'Warning: Lowest energy solution contains {len(LEaction)} actions. Review constraint conditions.')
        if LEaction==[]:
            # Choose scan if no action selected
            LEaction = self.nActions-1
        else:
            # Choose first action if multiple returned
            LEaction = LEaction[0] - self.nHidden
        return LEaction, list(LEsamp[0].values()) # Return LE action and state of hidden nodes as array

    def updateQval(self,state,action,Q):
        # Store calculated Q value
        if not type(Q)==list: # list if Q==[]
            self.Qvals[self.getStateI(state)][action] = Q
        return

    def getStateI(self,state):
        # Return indexed state value
        if state[-1]==1 or not self.storeQ:
            stateI=0
        else:
            stateI = round(np.inner(self.stateMult,state))
        return stateI

    def updateWeights(self,reward,Q1,Q2,state,action,h,state_,action_):
        observation = np.concatenate((state,action))


        if self.adaptiveGradient and self.outerGradSum==[]:
            self.outerGradSum = [np.zeros_like(self.hvWeights), np.zeros_like(self.hhWeights)]


        hvGradient = (reward + self.gamma*Q2 - Q1)*np.outer(observation,np.diag(h)) # Technically -gradient
        hvGradient = hvGradient * self.nonzeroHV
        if self.adaptiveGradient:
            # If applying adaptive gradients, calculate scale
            self.outerGradSum[0] += hvGradient**2
            outerGradSum_ = self.outerGradSum[0] 
            outerGradSum_[outerGradSum_==0] = 1
            gradScale = 1/np.sqrt(outerGradSum_) + (self.epsilon0/self.epsilon)
        else:
            gradScale = 1
        self.hvWeights += self.epsilon * hvGradient * gradScale

        hhGradient = (reward + self.gamma*Q2 - Q1)*h # Technically -gradient
        hhGradient = hhGradient * self.nonzeroHH
        if self.adaptiveGradient:
            # If applying adaptive gradients, calculate scale
            self.outerGradSum[1] += hhGradient**2
            outerGradSum_ = self.outerGradSum[1] 
            outerGradSum_[outerGradSum_==0] = 1
            gradScale = 1/np.sqrt(outerGradSum_) + (self.epsilon0/self.epsilon)
        else:
            gradScale = 1
        self.hhWeights += self.epsilon * hhGradient * gradScale

    def scaleReward(self,reward,direction='forward',Q=False):
        # Scale/unscale reward values
        if Q:
            scale = 100 * (1-self.gamma)
        else:
            scale = 100
        offset = 100
        if direction=='forward':
            # Reward -> scaled reward
            reward = reward + offset
            reward = reward / scale
        elif direction=='backward':
            reward = reward * scale
            reward = reward - offset
        return reward

    def AmendZeroState(self,state):
        if all(state==0):
            state = np.concatenate((state,[1]))
        else:
            state = np.concatenate((state,[0]))
        return state

    def ChooseAction(self,state,randomAction,actionI=[]):
        # Choose an action based on the current state
        # If randomAction==True, pick a random action with probability self.pRandom

        Q = []
        h = []
        if randomAction and (np.random.rand(1) <= max(self.pRandom,self.minPrandom)):
            actionI = np.random.randint(self.nActions)
        elif actionI==[]:
            actionI, Q, h = self.getOptimalAction(state)
        action = np.zeros(self.nActions)
        action[actionI] = 1
        if Q == [] and self.batchSize is None:
            Q, h = self.evaluateQBM(state,action,self.SimulateAnneal)
        elif not self.batchSize is None:
            Q = []
            h = []
        return Q, action, actionI, h

    def learn(self,nSteps = 2000):
        if not self.QBMinitialised:
            self.initRBM(5) # 5 Node RBM by default
        
        done = True
        state2 = self.AmendZeroState(self.env.reset())
        actionI2 = []
        nSteps = int(nSteps)
        self.log.initNsteps(self,nSteps)
        for self.step in range(1,(nSteps+1)):
            # Render

            if self.log.QPUseconds >= self.maximumQPUminutes * 60 :
                break
            # Get observation
            if done:
                self.gameReward = 0
                self.gameSteps = 0
                # if self.log.nGames % 10 == 0:
                #     self.render()
                
            state1 = state2

            # Choose an action
            Q1, action1, actionI1, h1 = self.ChooseAction(state1,randomAction = True,actionI=actionI2)
            # Update probability of taking a random step
            if self.step > self.nRandomSteps:
                self.pRandom = self.pRandom * self.pRandomDecay

            # Do action, returns reward and new state
            state2, reward, done, notes = self.env.step(actionI1)
            # if self.log.nGames % 10 == 0:
            #     self.render()
            if done:
                state2 = self.env.reset()

            scaledReward = self.scaleReward(reward)
            state2 = self.AmendZeroState(state2)

            # Choose an action2, calculate Q2
            Q2, action2, actionI2, h2 = self.ChooseAction(state2,randomAction = False)

            # Update weights and Q
            self.updateWeights(scaledReward,Q1,Q2,state1,action1,h1,state2,action2)
            if self.storeQ:
                self.updateQval(state1,actionI1,Q1)
                self.updateQval(state2,actionI2,Q2)

            # Logging
            self.gameReward += reward
            self.gameSteps  += 1
            self.log.update(self,state1,action1,reward,Q1,Q2,done)

        self.log.tidy()
        return self.log

    def exportResults(self,writeTables:bool=True,showFigs:bool=False,saveFigs:bool=True,storeMeta:bool=True):
        # Output results
        results = QBMResults(self)
        if writeTables:
            results.toExcel()
        if showFigs or saveFigs:
            results.plotAll(showFigs=showFigs,saveFigs=saveFigs)
        if storeMeta:
            results.saveMetadata()

    def loadEmbedding(self,index:int=0,fileName:str=''):
        # Load minor embedding for problem if it is already saved within the run folder
        self.embeddingLoaded[index] = True
        with open(fileName, 'r') as file:
            embedding = json.loads(file.read())
        embedding_ = {int(key):embedding[key] for key in list(embedding.keys())}
        self.quantumSampler[index] = FixedEmbeddingComposite(DWaveSampler(),embedding=embedding_)
        self.embeddingLoaded[index] = True

    def saveEmbedding(self,index:int=0,fileName:str=''):
        # Save calculated minor embedding for problem within the run folder
        if not os.path.isdir(os.path.dirname(fileName)):
            os.mkdir(os.path.dirname(fileName))
        embedding = self.quantumSampler[index].embedding

        with open(fileName, 'w') as file:
            file.write(json.dumps(embedding))

    def render(self):
        resultsFol = self.log.resultsDir+'\\Game_'+str(self.log.nGames+1)
        if not os.path.isdir(resultsFol):
            os.mkdir(resultsFol)
        self.env.render()
        self.env.graph_plotter.fig.savefig(resultsFol + \
                                    '\\Step_'+str(self.env.current_duration)+'.png')


