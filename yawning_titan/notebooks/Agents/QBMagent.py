import numpy as np
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan import _YT_ROOT_DIR

# import DWAVE model and sampler
import dimod
import os
# try:
#     from dwave.system import LeapHybridSampler
# except:
#     pass
from dwave.samplers import SimulatedAnnealingSampler

class QBMAgent:
#   QBMAgent - An agent that implements a Boltzmann machine (either Reduced BM or Deep BM) based reinforcement learning.
#   Functions:
#   initRBM             - Initialises a reduced boltzmann machine
#   initDBM             - Initialises a deep boltzmann machine
#   loadWeights         - Initialises a generic BM based on saved weight .txt files. Assumes any zero weights are unconnected nodes
#   buildHamiltonian    - Builds a hamiltonian to represent the free energy in the machine, either state and action, or state only are acceptable inputs
#   calculateFreeEnergy - Calculates the free energy from the hamiltonian and samples of the hamiltonian
#   evaluateQBM         - Evaluates the boltzmann machine - builds hamiltonian and calculates the energy
#   getOptimalAction    - Find the optimal action for a given state. Option to sample BM for lowest energy, or directly choose from saved Q values
#   getLowEnergyAction  - Find the action that has the lowest energy from a sampled hamiltonian (with action nodes free)
#   updateQval          - Update the stored Q value for a state, action pair
#   getStateI           - Get the index of a given state for use in the Q matrix
#   updateWeights       - Update the weights of the BM
#   scaleReward         - Scale the reward (forward or backward), useful to improve BM fitting
#   AmendZeroState      - Amend a state of all zeros to a state of all zeros, with a 1 final node flag
#   ChooseAction        - Choose an action and evaluate the Q value from the BM
#   learn               - Run the training loop
#   updateLog           - Log the current training step

    def __init__(self,env: GenericNetworkEnv):
        self.env = env

        # Get number of actions/observations
        self.nActions = env.blue_actions
        self.nObservations = env.observation_space.shape[0] + 1
        self.nVisible = self.nActions + self.nObservations

        # Learning hyperparameters
        self.beta = 5 # Thermodynamic beta in free energy calculations (also used in simulated annealing)
        self.epsilon = 1e-1 # Update step length
        self.epsilon0 = 1e-3 # For use with adaptive gradient - minimum epsilon
        self.gamma = 0.9 # Weight on future reward

        # Random action choice parameters
        self.nRandomSteps = 1000 # Number of random steps to take in initial learning
        self.pRandomDecay = 0.99 # Stepwise decay on p(take random step) after nRandomSteps
        self.pRandom = 1 # Initial p(take random step) value
        self.minPrandom = 0.01 # Minimum p(take random step) value
        
        # Logging
        self.LogRate = 200
        self.log = {"state": [],
            "action": [],
            "Reward": [],
            "ExpectedReward": [],
            "ExpectedReward0": [],
            "ExpectedReward1": [],
            "GameAvgReward":[],
            "TotAvgReward":[],
            "Qerror": [],
            "GameLength":[],
            "GameReward":[]}

        # Internal counters
        self.steps = 0
        self.outerGradSum = []

        # Set flags
        self.QBMinitialised = False # Flag to check nodes are set up
        self.SimulateAnneal = True # Simulate or use D-Wave
        self.adaptiveGradient = True # Flag to use adaptive gradient method
        self.AnnealToBestAction = True # Flag to choose best action via annealing rather than reviewing Q values

        # Augment sample options
        # For each sampled state, produce {AugmentScale} copies of the state, where each node is switched
        # with probability {pSwitch}. This helps to improve the estimated values of <H> and <h>.
        self.AugmentSamples = True
        self.AugmentScale = 50
        self.pSwitch = 0.2 # Probability of each individual node switching
      
        # Set Stored Q maximums
        self.storeQ = self.nObservations<30
        if not self.AnnealToBestAction and not self.storeQ:
            print('Warning: too many observables to store Q values. ''AnnealToBestAction'' flag has been set to ''True''')
            self.AnnealToBestAction = True
        if self.storeQ:
            self.Qvals = np.ones(shape=(2**(self.nObservations-1),self.nActions))*np.nan
            self.stateMult = [2**i for i in range(self.nObservations)]
  
    def initRBM(self,hiddenNodes = 5):
        # Initialise an RBM with random weights, zero weight between hidden nodes
        self.nHidden = hiddenNodes
        self.hvWeights = np.random.randn(self.nVisible,self.nHidden)
        self.hhWeights = np.zeros((self.nHidden,self.nHidden))

        self.nonzeroHV = np.ones_like(self.hvWeights)
        self.nonzeroHH = np.zeros_like(self.hhWeights)

        self.QBMinitialised = True

    def initDBM(self,hiddenNodes = [5,5]):
        # Initialise an DBM with random weights
        self.nHidden = sum(hiddenNodes)
        self.hvWeights = np.random.randn(self.nVisible,self.nHidden)
        self.hhWeights = np.random.randn(self.nHidden,self.nHidden)

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
        self.hvWeights = os.path.join(resultsDir,'hvWeights.txt')
        self.hhWeights = os.path.join(resultsDir,'hhWeights.txt')

        self.nHidden = np.shape(self.hvWeights)[0]
        self.nonzeroHV = (self.hvWeights != 0).astype(int)
        self.nonzeroHH = (self.hhWeights != 0).astype(int)
        self.QBMinitialised = True

    def saveWeights(self):
        np.savetxt(os.path.join(self.ResultsDir,'hvWeights.txt'),self.agent.hvWeights,delimiter=',')
        np.savetxt(os.path.join(self.ResultsDir,'hhWeights.txt'),self.agent.hhWeights,delimiter=',')


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
        Hamiltonian = dimod.BinaryQuadraticModel(nFreeNodes, 'BINARY')
        # Build Hamiltonian
        for iH in range(self.nHidden):
        # Loop over nodes
            for iV in range(self.nVisible):
                if iV<self.nObservations or not freeAction:
                    # Interactions with observations
                    Hamiltonian.linear[iH] -= self.hvWeights[iV,iH] * observation[iV]
                else:
                    Hamiltonian.add_quadratic(iH,self.nHidden+iV-self.nObservations,-self.hvWeights[iV,iH])
            for iH2 in range(iH+1,self.nHidden):
            # Interactions with other hidden nodes
                if self.nonzeroHH[iH,iH2] == 1:
                    Hamiltonian.add_quadratic(iH,iH2,-self.hhWeights[iH,iH2])
        if freeAction:
            # Solver not working for CQM
            # Hamiltonian = dimod.ConstrainedQuadraticModel.from_bqm(Hamiltonian)
            # Hamiltonian.add_constraint_from_iterable([(iV,1) for iV in range(self.nHidden,nFreeNodes)],'==',rhs=1)

            # Add penalty term to ensure one node is activated
            penaltyScale = 10
            indOffset = self.nHidden-self.nObservations
            for iV in range(self.nObservations,self.nVisible):
                Hamiltonian.linear[iV+indOffset] -= penaltyScale
                for iV2 in range(iV+1,self.nVisible):
                    Hamiltonian.add_quadratic(iV+indOffset,iV2+indOffset,2 * penaltyScale)
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
                for iAdd in range(self.AugmentScale):
                    # Probability for getting all nodes to switch at least once is
                    # (1-(1-pSwitch)^AugmentScale)^nHidden
                    flipBit = np.random.rand(self.nHidden)<self.pSwitch 
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
        for record in resAgg:
            Zv += np.exp(-record[1] * beta)

        # For each record, calculate probability, mean hamiltonian contribution,
        # entropy contribution and h interaction contributions
        for record in resAgg:
            pEnergy = np.exp(-record[1] * beta)/Zv

            HamAvg += record[1] * pEnergy
            entropy += pEnergy * np.log(pEnergy)
            for iH in range(self.nHidden):
                h[iH,iH] += pEnergy * record[0][iH]
                for iH2 in range(iH+1,self.nHidden):
                    h[iH,iH2] += pEnergy * record[0][iH] * record[0][iH2]

        minusF = - HamAvg - 1/beta * entropy
        return minusF, h

    def evaluateQBM(self,state,action,SimulateAnneal=True):
        # This function receives the current state and action choice
        # It then calculates the value of -F, and the average h values for use in weight updates
        # TODO: Add option to explicitly evaluate RBM without sampling

        # Build Hamiltonian
        Hamiltonian = self.buildHamiltonian(state,action)

        # Sample Hamiltonian and aggregate results
        if SimulateAnneal:
            beta0 = min(0.1,self.beta/5)
            sampler = SimulatedAnnealingSampler()
            results = sampler.sample(Hamiltonian,num_reads=10,beta_range=[beta0, self.beta])
        else:
            sampler = LeapHybridSampler() 
            sampler.sample(Hamiltonian)
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

    def getOptimalAction(self,state):
        stateI = self.getStateI(state)

        if self.AnnealToBestAction:
            # Sample Hamiltonian to get the action with the lowest energy (highest reward)
            actionI, Q, h = self.evaluateQBM(state,[],self.SimulateAnneal)
        elif any(np.isnan(self.Qvals[stateI])):
            # Haven't evaluated Q for all actions yet
            Q = self.Qvals[stateI]
            hA = [[]]*self.nActions
            # Loop over actions, calculate Q for those without a stored value
            for iA in range(self.nActions):
                if np.isnan(Q[iA]):
                    action = np.zeros(self.nActions)
                    action[iA] = 1
                    Q[iA], hA[iA] = self.evaluateQBM(state,action,self.SimulateAnneal)

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
        self.Qvals[self.getStateI(state)][action] = Q
        return

    def getStateI(self,state):
        # Return indexed state value
        if state[-1]==1 or not self.storeQ:
            stateI=0
        else:
            stateI = round(np.inner(self.stateMult,state))
        return stateI

    def updateWeights(self,reward,Q1,Q2,state,action,h):
        observation = np.concatenate((state,action))


        if self.adaptiveGradient and self.outerGradSum==[]:
            self.outerGradSum = [np.zeros_like(self.hvWeights), np.zeros_like(self.hhWeights)]


        for iH in range(self.nHidden):
            # Loop over hidden nodes
            for iV in range(self.nVisible):
                # Loop over visible nodes
                if self.nonzeroHV[iV,iH]==1:
                    # If connected, calculate gradient and update weight
                    gradient = (reward + self.gamma*Q2 - Q1)*observation[iV]*h[iH,iH] # Technically -gradient
                    if self.adaptiveGradient:
                        # If applying adaptive gradients, calculate scale
                        if gradient==0:
                            gradScale=1 # Avoid divide by zero error
                        else:
                            self.outerGradSum[0][iV,iH] += gradient**2
                            gradScale = 1/np.sqrt(self.outerGradSum[0][iV,iH]) + (self.epsilon0/self.epsilon)
                    else:
                        gradScale = 1
                    self.hvWeights[iV,iH] += self.epsilon * gradient * gradScale
                    
            for iH2 in range(iH+1,self.nHidden):
                # Loop over other hidden nodes
                if self.nonzeroHH[iH,iH2]==1:
                    # If connected, calculate gradient and update weight
                    gradient = (reward + self.gamma*Q2 - Q1)*h[iH,iH2] # Technically -gradient
                    if self.adaptiveGradient:
                        # If applying adaptive gradients, calculate scale
                        if gradient==0:
                            gradScale=1
                        else:
                            self.outerGradSum[1][iH,iH2] += gradient**2
                            gradScale = 1/np.sqrt(self.outerGradSum[1][iH,iH2]) + (self.epsilon0/self.epsilon)
                    else:
                        gradScale = 1
                    self.hhWeights[iH,iH2] += self.epsilon * gradient * gradScale
                    
    def scaleReward(self,reward,direction='forward',Q=False):
        # Scale/unscale reward values
        if Q:
            scale = 10 * (1-self.gamma)
        else:
            scale = 10
        offset = 20
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

    def ChooseAction(self,state,randomAction):
        # Choose an action based on the current state
        # If randomAction==True, pick a random action with probability self.pRandom

        if randomAction and (np.random.rand(1) <= max(self.pRandom,self.minPrandom)):
            actionI = np.random.randint(self.nActions)
            Q = []
        else:
            actionI, Q, h = self.getOptimalAction(state)
        action = np.zeros(self.nActions)
        action[actionI] = 1
        if Q == []:
            Q, h = self.evaluateQBM(state,action,self.SimulateAnneal)
        return Q, action, actionI, h

    def learn(self,nSteps = 2000):
        if not self.QBMinitialised:
            self.initRBM(5) # 5 Node RBM by default
        
        done = True
        for iStep in range(nSteps):
            # Get observation
            if done:
                state1 = self.env.reset()
                self.gameReward = 0
                self.gameSteps = 0
                state1 = self.AmendZeroState(state1)
            else:
                state1 = state2

            # Update probability of taking a random step
            if iStep > self.nRandomSteps:
                self.pRandom = self.pRandom * self.pRandomDecay
            # Choose an action
            Q1, action1, actionI1, h1 = self.ChooseAction(state1,randomAction = True)

            # Do action, returns reward and new state
            state2, reward, done, notes = self.env.step(actionI1)
            reward = self.scaleReward(reward)
            state2 = self.AmendZeroState(state2)

            # Choose an action2, calculate Q2
            Q2, action2, actionI2, _ = self.ChooseAction(state1,randomAction = False)

            # Update weights and Q
            self.updateWeights(reward,Q1,Q2,state1,action1,h1)
            if self.storeQ:
                self.updateQval(state1,actionI1,Q1)
                self.updateQval(state2,actionI2,Q2)

            # Logging
            self.updateLog(iStep,state1,action1,reward,Q1,Q2,done)
        return self.log

    def updateLog(self,step,state,action,reward,Q1,Q2,done):
        reward = self.scaleReward(reward,'backward')
        self.gameReward += reward
        self.gameSteps  += 1


        self.log["state"]  += [state]
        self.log["action"] += [action]
        if self.storeQ:
            self.log["ExpectedReward"] += [self.scaleReward(self.Qvals[self.getStateI(state)].copy(),'backward',True)]
            self.log["ExpectedReward0"] += [self.scaleReward(self.Qvals[0].copy(),'backward',True)]
            self.log["ExpectedReward1"] += [self.scaleReward(self.Qvals[1].copy(),'backward',True)]
        self.log["Reward"] += [reward]
        self.log["GameAvgReward"] += [self.gameReward/self.gameSteps]
        self.log["TotAvgReward"] += [np.sum(self.log["Reward"])/(step+1)]
        self.log["Qerror"] += [reward - self.scaleReward(-self.gamma*Q2 + Q1,'backward')]

        if self.LogRate>0 and np.mod(step+1,self.LogRate)==0:
            print('')
            print('Step '+str(step+1))
            print('Total Average Reward = '+str(self.log["TotAvgReward"][-1]))
            if self.storeQ:
                print('E[s0] = '+str(self.scaleReward(self.Qvals[0].copy(),'backward',True)))
        
        if done:
            self.log["GameLength"] += [self.gameSteps]
            self.log["GameReward"] += [self.gameReward]
        return
