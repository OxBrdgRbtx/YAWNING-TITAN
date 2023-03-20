import numpy as np
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.notebooks.Outputs.qbmLogger import qbmLogger
from yawning_titan.notebooks.Outputs.QBMResults import QBMResults

# import DWAVE model and sampler
from dimod import BinaryQuadraticModel
import os
from dwave.system import LazyFixedEmbeddingComposite, DWaveSampler
from dimod import SimulatedAnnealingSampler

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
        AugmentSamples:bool=True,AugmentScale:int=100,augmentPswitch:float=0.1): # Sample Augmentation options


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
                self.quantumSampler = [LazyFixedEmbeddingComposite(DWaveSampler())]
        else:
            # (not SimulateAnneal) AND (not SimulateAnnealForAction)
            # Need two different embeddings
            self.quantumSampler = [LazyFixedEmbeddingComposite(DWaveSampler()), LazyFixedEmbeddingComposite(DWaveSampler())]

        # Augment sample options
        # For each sampled state, produce {AugmentScale} copies of the state, where each node is switched
        # with probability {augmentPswitch}. This helps to improve the estimated values of <H> and <h>.
        self.AugmentSamples = AugmentSamples
        self.AugmentScale = AugmentScale
        self.augmentPswitch = augmentPswitch# Probability of each individual node switching
      
        # Set Stored Q maximums
        self.storeQ = self.nObservations<22
        if not self.AnnealToBestAction and not self.storeQ:
            print('Warning: too many observables to store Q values. ''AnnealToBestAction'' flag has been set to ''True''')
            self.AnnealToBestAction = True
        if self.storeQ:
            self.Qvals = np.ones(shape=(2**(self.nObservations-1),self.nActions))*np.nan
            self.stateMult = [2**i for i in range(self.nObservations)]
  
    def initRBM(self,hiddenNodes = 5):
        # Initialise an RBM with random weights, zero weight between hidden nodes
        self.nHidden = hiddenNodes
        self.hvWeights = np.random.normal(loc=0.0,scale=1e-3,size=(self.nVisible,self.nHidden))
        self.hhWeights = np.zeros((self.nHidden,self.nHidden))

        self.nonzeroHV = np.ones_like(self.hvWeights)
        self.nonzeroHH = np.zeros_like(self.hhWeights)

        self.QBMinitialised = True

    def initDBM(self,hiddenNodes = [5,5]):
        # Initialise an DBM with random weights
        self.nHidden = sum(hiddenNodes)
        self.hvWeights = np.random.normal(loc=0.0,scale=1e-3,size=(self.nVisible,self.nHidden))
        self.hhWeights = np.random.normal(loc=0.0,scale=1e-3,size=(self.nHidden,self.nHidden))

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
            Hamiltonian.add_linear_from_array(-np.matmul(np.transpose(self.hvWeights),observation))
            Hamiltonian.add_quadratic_from_dense(quadTerms)
        else:
            linearTerms = np.zeros((nFreeNodes,))
            linearTerms[:self.nHidden] = -np.matmul(np.transpose(self.hvWeights[:self.nObservations,:]),observation)
            quadTerms[self.nHidden:,:self.nHidden] = -self.hvWeights[self.nObservations:,:]
            
            # Add penalty term to ensure one node is activated
            penaltyScale = 20
            linearTerms[self.nHidden:] = -penaltyScale
            quadTerms[self.nHidden:,self.nHidden:] = np.triu(2 * penaltyScale * np.ones((self.nActions,self.nActions)),1)

            Hamiltonian.add_linear_from_array(linearTerms)
            Hamiltonian.add_quadratic_from_dense(quadTerms)

        return Hamiltonian

    def calculateFreeEnergy(self,results,beta,Hamiltonian):
        # Take results from sampled hamiltonian and calculate mean free energy

        # Add additional samples near the minima to help in calculating mean
        if self.AugmentSamples:
            if type(results) is list:
                records = [[results,Hamiltonian.energy(results)]]
            else:
                resAgg0 = results.aggregate()
                records = resAgg0.record
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

                    # Then flip pairs of bits
                    for iFlip2 in range(iFlip+1,self.nHidden):
                        flipBit = np.zeros(self.nHidden)
                        flipBit[iFlip] = 1
                        flipBit[iFlip2] = 1
                        thisSamp = np.mod(record[0] + flipBit,2)
                        thisEnergy = Hamiltonian.energy(thisSamp)
                        resAug += [[thisSamp,thisEnergy]]
                        sampAgg += [thisSamp]
                
            _,uInds = np.unique(np.array(sampAgg),axis=0,return_index=True)
            resAgg = [resAug[iU] for iU in uInds]
        else:
            if type(results) is list:
                resAgg = [[results,Hamiltonian.energy(results)]]
            else:
                resAgg = results.aggregate().record

        # Calculate Zv
        HamAvg = 0.0
        entropy = 0.0
        Zv = 0.0
        h = np.zeros((self.nHidden,self.nHidden))
        minEnergy = min([record[1] for record in resAgg]) # Stop overflowing exp
        for record in resAgg:
            Zv += np.exp(-(record[1]-minEnergy) * beta)

        # For each record, calculate probability, mean hamiltonian contribution,
        # entropy contribution and h interaction contributions
        for record in resAgg:
            pEnergy = np.exp(-(record[1]-minEnergy) * beta)/Zv

            HamAvg += record[1] * pEnergy
            entropy += pEnergy * np.log(pEnergy)
            for iH in range(self.nHidden):
                h[iH,iH] += pEnergy * record[0][iH]
                for iH2 in range(iH+1,self.nHidden):
                    h[iH,iH2] += pEnergy * record[0][iH] * record[0][iH2]

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
        nSamples = len(results.record)

        if action == []:
            # No action supplied, searching for the best choice of action
            actionI,resultSample = self.getLowEnergyAction(results)
            Hamiltonian = self.buildHamiltonian(state,resultSample[self.nHidden:])
            minusF, h = self.calculateFreeEnergy(resultSample[:self.nHidden],self.beta,Hamiltonian)
            return actionI, minusF, h
        else:
            # Process results and calculate mean energy, h
            minusF, h = self.calculateFreeEnergy(results,self.beta,Hamiltonian)
            return minusF, h

    def sampleHamiltonian(self,Hamiltonian,SimulateAnneal):
        if SimulateAnneal:
            beta0 = min(0.1,self.SAbeta/5)
            results = self.simulatedSampler.sample(Hamiltonian,num_reads=10,beta_range=[beta0, self.SAbeta])
        else:
            nHidden = self.nHidden
            if self.batchSize is not None:
                nHidden = self.nHidden * 2 * self.batchSize
            if len(Hamiltonian) == nHidden:
                iSampler = 0
            else:
                iSampler = 1 # If sampling for action, need two separate fixed embeddings
            try:
                results = self.quantumSampler[iSampler].sample(Hamiltonian,num_reads=100)
                self.log.addQPUtime(results.info["timing"]["qpu_access_time"])
            except:
                beta0 = min(0.1,self.SAbeta/5)
                results = self.simulatedSampler.sample(Hamiltonian,num_reads=100,beta_range=[beta0, self.SAbeta])
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
            # Get observation
            if done:
                self.gameReward = 0
                self.gameSteps = 0
                
            state1 = state2

            # Choose an action
            Q1, action1, actionI1, h1 = self.ChooseAction(state1,randomAction = True,actionI=actionI2)
            # Update probability of taking a random step
            if self.step > self.nRandomSteps:
                self.pRandom = self.pRandom * self.pRandomDecay

            # Do action, returns reward and new state
            state2, reward, done, notes = self.env.step(actionI1)
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