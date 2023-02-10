import os
import numpy as np
import time
from yawning_titan import _YT_ROOT_DIR


class qbmLogger:
    def __init__(self,saveName: str,printRate:int=int(1e4),gameWindow:int=10,stepWindow:int=1000,
    writeStepLogs:bool=False,writeGameLogs:bool=False,writeWeights:bool=False,writeToTerminal:bool=True):
        
        # Set up results directory
        self.resultsDir = os.path.join(_YT_ROOT_DIR,'results',saveName)
        if not os.path.isdir(self.resultsDir) and (writeStepLogs or writeGameLogs or writeWeights):
            os.makedirs(self.resultsDir)
        
        # Store input options
        self.printRate = printRate
        self.gameWindow = gameWindow
        self.stepWindow = stepWindow
        self.writeStepLogs = writeStepLogs
        self.writeGameLogs = writeGameLogs
        self.writeWeights = writeWeights
        self.writeToTerminal = writeToTerminal
        
        self.newStepLog = True
        self.newGameLog = True

        # Set up log dicts
        self.gameLog = {"Length":[],
            "Reward":[],
            "Average Reward":[],
            "Rolling Average Length":[],
            "Rolling Average Reward":[]}
        self.nGames = 0

        self.stepLog = {"state": [],
            "action": [],
            "Reward": [],
            "Expected Reward": [],
            "Average Reward": [],
            "Rolling Average Reward": [],
            "Qerror": [],
            "Q0":[]}

        # Set up profiling
        self.tStart = time.time()
        self.t0 = time.time()

    def initNsteps(self,agent,nSteps):
        # Initialise arrays for each log for speed
        for key in self.gameLog.keys():
            self.gameLog[key] = np.empty((nSteps,))
        for key in self.stepLog.keys():
            self.stepLog[key] = np.empty((nSteps,))
        self.stepLog["state"] = np.empty((nSteps,agent.nObservations))
        self.stepLog["action"] = np.empty((nSteps,agent.nActions))
        self.stepLog["Expected Reward"] = np.empty((nSteps,agent.nActions))
        self.stepLog["Q0"] = np.empty((nSteps,agent.nActions))

    def tidy(self):
        self.tEnd = time.time()
        for key in self.gameLog.keys():
            self.gameLog[key] = self.gameLog[key][:self.nGames]


    def update(self,agent,state,action,reward,Q1,Q2,done):
        self.step = agent.step-1
        self.updateStepLog(agent,state,action,reward,Q1,Q2)
        if done:
            self.updateGameLog(agent)
            if self.writeGameLogs:
                self.gameLogtoTxt(agent)

        if np.mod(agent.step,self.printRate)==0:
            if self.writeStepLogs:
                self.stepLogtoTxt(agent)
            if self.writeWeights:
                self.weightsToTxt(agent)
            if self.writeToTerminal:
                self.printToTerminal(agent)

    def updateStepLog(self,agent,state,action,reward,Q1,Q2):
        self.stepLog["state"][self.step,:] = state
        self.stepLog["action"][self.step,:] = action

        if agent.storeQ:
            self.stepLog["Expected Reward"][self.step,:] = agent.scaleReward(agent.Qvals[agent.getStateI(state)].copy(),'backward',True)
            self.stepLog["Q0"][self.step,:] = agent.scaleReward(agent.Qvals[0].copy(),'backward',True)
        self.stepLog["Reward"][self.step] = reward
        self.stepLog["Average Reward"][self.step] = np.sum(self.stepLog["Reward"])/(agent.step)
        self.stepLog["Qerror"][self.step] = reward - agent.scaleReward(-agent.gamma*Q2 + Q1,'backward')

        minStep = max((0,agent.step-self.stepWindow))
        self.stepLog["Rolling Average Reward"][self.step] = np.sum(self.stepLog["Reward"][minStep:agent.step])/(agent.step-minStep)

    def updateGameLog(self,agent):
        nGames = self.nGames
        self.nGames += 1
        self.gameLog["Length"][nGames] = agent.gameSteps
        self.gameLog["Reward"][nGames] = agent.gameReward
        self.gameLog["Average Reward"][nGames] = agent.gameReward/agent.gameSteps

        self.getRollingAverageGame()
        self.gameLog["Rolling Average Length"][nGames] = self.rollingAverageGameLength
        self.gameLog["Rolling Average Reward"][nGames] = self.rollingAverageGameReward
    
    def stepLogtoTxt(self,agent):
        logFile = os.path.join(self.resultsDir,'StepInfo.csv')
        try:
            if self.newStepLog:
                with open(logFile,'w') as f:
                    f.write(f"Step, Total Mean Reward, Previous {self.stepWindow} steps average reward, Previous {self.gameWindow} games mean length, Previous {self.gameWindow} games mean reward\n")
                self.newStepLog = False
            with open(logFile,"ab") as f:
                np.savetxt(f,np.array([agent.step, self.stepLog["Average Reward"][self.step], self.stepLog["Rolling Average Reward"][self.step], self.rollingAverageGameLength, self.rollingAverageGameReward]).reshape((1,-1)),delimiter=',')
        except:
            print(f'Failed writing step log to StepInfo.csv')
    
    def gameLogtoTxt(self,agent):
        logFile = os.path.join(self.resultsDir,'GameInfo.csv')
        try:
            if self.newGameLog:
                with open(logFile,'w') as f:
                    f.write(f"Game Number, Length, Reward, Average Reward, Rolling Average Length ({self.gameWindow} games), Rolling Average Reward ({self.gameWindow} games)\n")
                self.newGameLog = False
            with open(logFile,"ab") as f:
                np.savetxt(f,np.array([self.nGames, agent.gameSteps, agent.gameReward, 
                    agent.gameReward/agent.gameSteps, self.rollingAverageGameLength, 
                    self.rollingAverageGameReward]).reshape((1,-1)),delimiter=',')
        except:
            print(f'Failed writing step log to GameInfo.csv')

    def weightsToTxt(self,agent):
        try:
            agent.saveWeights(os.path.join(self.resultsDir,f'Step_{agent.step}'))
        except:
            print(f'Failed saving weights')

    def printToTerminal(self,agent):
        # Do some profiling first
        t1 = time.time()
        totalTime = t1 - self.tStart
        periodTime = t1 - self.t0
        self.t0 = t1

        print('')
        print('Step '+str(agent.step))
        print(f'{totalTime:.1f} seconds elapsed ({periodTime:.1f}s since previous update)')
        print(f'{1000*totalTime/agent.step:.1f}ms/step ({1000*periodTime/self.printRate:.1f}ms/step since previous update)')
        print(f'Total Average Reward = {self.stepLog["Average Reward"][self.step]}')
        if self.nGames>0:
            window = self.thisGameWindow
            aLength = [self.rollingAverageGameLength]+self.rollingAverageGameLength_lims
            aReward = [self.rollingAverageGameReward]+self.rollingAverageGameReward_lims
            print(f'Last {window} games average length ([min, max]): {aLength[0]:.3f} ([{aLength[1]}, {aLength[2]}])')
            print(f'Last {window} games average reward/step ([min, max]): {aReward[0]:.3f}  ([{aReward[1]:.3f}, {aReward[2]:.3f}])')
        if agent.storeQ:
            print(f'Q0 = {self.stepLog["Q0"][self.step]}')

    def getRollingAverageGame(self):
        if self.nGames>0:
            if self.nGames<self.gameWindow:
                self.thisGameWindow = self.nGames
            else:
                self.thisGameWindow = self.gameWindow
            minStep = self.nGames-self.thisGameWindow
            thisStep = self.nGames
            lengths = self.gameLog["Length"][minStep:thisStep]
            rewards = self.gameLog["Reward"][minStep:thisStep]
            avgRewards = self.gameLog["Average Reward"][minStep:thisStep]
            self.rollingAverageGameLength = np.mean(lengths)
            self.rollingAverageGameLength_lims = [np.min(lengths),np.max(lengths)]
            self.rollingAverageGameReward = np.sum(rewards)/(self.rollingAverageGameLength*self.thisGameWindow)
            self.rollingAverageGameReward_lims = [np.min(avgRewards),np.max(avgRewards)]

