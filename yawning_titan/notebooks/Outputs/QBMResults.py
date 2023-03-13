import pandas as pd
import numpy as np
import os
from matplotlib import pyplot as plt
import yaml

class QBMResults:
    def __init__(self,Agent):
        self.agent = Agent
        self.log = Agent.log
        self.resultsDir = Agent.log.resultsDir
        if not os.path.isdir(self.resultsDir):
            os.makedirs(self.resultsDir)
            
    def toExcel(self):
        logs = {"stepLog": self.log.stepLog,
            "gameLog": self.log.gameLog}
        for thisLog in logs.keys():
            try:
                file = os.path.join(self.resultsDir,thisLog+'.xlsx')
                with pd.ExcelWriter(file, engine='openpyxl', mode='w+') as writer:  
                    for key in list(logs[thisLog].keys()):
                        df = pd.DataFrame(logs[thisLog][key])
                        df.to_excel(writer, sheet_name=key)
            except:
                for key in list(logs[thisLog].keys()):
                    df = pd.DataFrame(logs[thisLog][key])
                    df.to_csv(os.path.join(self.resultsDir,thisLog+'_'+key+'.csv'))

    def plotAll(self,showFigs = True,saveFigs = False,figSize=(7,4)):
        self.plotStepData(showFigs=showFigs,saveFigs=saveFigs,figSize=figSize)
        self.plotGameData(showFigs=showFigs,saveFigs=saveFigs,figSize=figSize)

    def plotStepData(self,keys = [],showFigs = True,saveFigs = False,figSize=(7,4)):
        self.plot(self.log.stepLog,"Step",keys=keys,showFigs=showFigs,saveFigs=saveFigs,figSize=figSize)

    def plotGameData(self,keys = [],showFigs = True,saveFigs = False,figSize=(7,4)):
        self.plot(self.log.gameLog,"Game",keys=keys,showFigs=showFigs,saveFigs=saveFigs,figSize=figSize)

    def plot(self,log,xlabel=[],keys = [],showFigs = True,saveFigs = False,figSize=(7,4)):
        if keys == []:
            keys = list(log.keys())
        for key in keys:
            df = pd.DataFrame(log[key])
            isDiscrete = self.isDiscrete(log,key)

            fig = plt.figure(figsize=figSize)
            xdata = range(len(df))
            for iC,col in enumerate(df.columns):
                if isDiscrete:
                    plotI = df[col]==1
                    thisx = np.array(xdata)
                    thisx = thisx[plotI]
                    plt.scatter(thisx,iC*np.ones_like(thisx))
                elif 'Average' in key:
                    plt.plot(xdata,df[col],label=str(iC))
                else:
                    plt.scatter(xdata,df[col],label=str(iC),s=2)
            plt.xlabel(xlabel)
            plt.ylabel(key)              
            if not isDiscrete:
                plt.legend()
            if saveFigs:
                plt.savefig(os.path.join(self.resultsDir,xlabel+'_'+key+'.png'))
            if showFigs:
                plt.show()
            else:
                plt.close(fig)

    def isDiscrete(self,log,key):
        df = pd.DataFrame(log[key])
        return np.all(np.logical_or(df==0,df ==1))
    
    def saveMetadata(self):
        # Save agent parameters
        fields = ["nActions", "nObservations", "nHidden", "beta", "epsilon", "epsilon0", 
            "gamma", "nRandomSteps", "pRandomDecay", "minPrandom",
            "SimulateAnneal", "adaptiveGradient", "AnnealToBestAction", "SimulateAnnealForAction",
            "explicitRBM", "AugmentSamples", "AugmentScale", "augmentPswitch","batchSize"]

        logfile = os.path.join(self.resultsDir,'Metadata.csv')
        with open(logfile,'w') as f:
            f.write(f"Runtime,{self.agent.log.tEnd-self.agent.log.tStart},\n")
            for field in fields:
                f.write(f"{field},{self.agent.__getattribute__(field)}\n")
          
        # Save game options
        gamefile = os.path.join(self.resultsDir,'gameMode.yml')
        with open(gamefile, "w") as file:
            yaml.safe_dump(self.agent.env.network_interface.settings, file)
        
        # Save weights
        self.agent.saveWeights(self.resultsDir)

        