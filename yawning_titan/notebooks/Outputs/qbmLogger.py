import os
import numpy as np
from yawning_titan import _YT_ROOT_DIR


class qbmLogger:
    def __init__(self,saveName: str,printRate:int=int(1e4),gameWindow:int=10,
    writeStepLogs:bool=False,writeGameLogs:bool=False,writeWeights:bool=False,writeToTerminal:bool=True):
        
        # Set up results directory
        self.resultsDir = os.path.join(_YT_ROOT_DIR,'results',saveName)
        if not os.path.isdir(self.resultsDir) and (writeStepLogs or writeGameLogs or writeWeights):
            os.makedirs(self.resultsDir)
        
        # Store input options
        self.printRate = printRate
        self.gameWindow = gameWindow
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
            "Qerror": [],
            "Q0":[]}

    def update(self,agent,state,action,reward,Q1,Q2,done):
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
        self.stepLog["state"]  += [state]
        self.stepLog["action"] += [action]

        if agent.storeQ:
            self.stepLog["Expected Reward"] += [agent.scaleReward(agent.Qvals[agent.getStateI(state)].copy(),'backward',True)]
            self.stepLog["Q0"] += [agent.scaleReward(agent.Qvals[0].copy(),'backward',True)]
        self.stepLog["Reward"] += [reward]
        self.stepLog["Average Reward"] += [np.sum(self.stepLog["Reward"])/(agent.step)]
        self.stepLog["Qerror"] += [reward - agent.scaleReward(-agent.gamma*Q2 + Q1,'backward')]

    def updateGameLog(self,agent):
        self.nGames += 1
        self.gameLog["Length"] += [agent.gameSteps]
        self.gameLog["Reward"] += [agent.gameReward]
        self.gameLog["Average Reward"] += [agent.gameReward/agent.gameSteps]

        self.getRollingAverageGame()
        self.gameLog["Rolling Average Length"] += [self.rollingAverageLength]
        self.gameLog["Rolling Average Reward"] += [self.rollingAverageReward]
    
    def stepLogtoTxt(self,agent):
        logFile = os.path.join(self.resultsDir,'StepInfo.csv')
        try:
            if self.newStepLog:
                with open(logFile,'w') as f:
                    f.write(f"Step, Total Mean Reward, Previous {self.gameWindow} games mean length, Previous {self.gameWindow} games mean reward\n")
                self.newStepLog = False
            with open(logFile,"ab") as f:
                np.savetxt(f,np.array([agent.step, self.stepLog["Average Reward"][-1], self.rollingAverageLength, self.rollingAverageReward]).reshape((1,-1)),delimiter=',')
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
                np.savetxt(f,np.array([self.nGames, agent.gameSteps, agent.gameReward, agent.gameReward/agent.gameSteps]).reshape((1,-1)),delimiter=',')
        except:
            print(f'Failed writing step log to GameInfo.csv')

    def weightsToTxt(self,agent):
        try:
            agent.saveWeights(os.path.join(self.resultsDir,f'Step_{agent.step}'))
        except:
            print(f'Failed saving weights')

    def printToTerminal(self,agent):
        print('')
        print('Step '+str(agent.step))
        print(f'Total Average Reward = {self.stepLog["Average Reward"][-1]}')
        if self.nGames>0:
            print(f'Last {self.thisGameWindow} games average length: {self.rollingAverageLength}')
            print(f'Last {self.thisGameWindow} games average reward: {self.rollingAverageReward}/step')
        if agent.storeQ:
            print(f'Q0 = {self.stepLog["Q0"][-1]}')

    def getRollingAverageGame(self):
        if self.nGames>0:
            if self.nGames<self.gameWindow:
                self.thisGameWindow = self.nGames
            else:
                self.thisGameWindow = self.gameWindow
            self.rollingAverageLength = np.mean(self.gameLog["Length"][-self.thisGameWindow:])
            self.rollingAverageReward = np.sum(self.gameLog["Reward"][-self.thisGameWindow:])/(self.rollingAverageLength*self.thisGameWindow)

