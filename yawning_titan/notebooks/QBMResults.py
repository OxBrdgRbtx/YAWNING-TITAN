import pandas as pd
import numpy as np
import os
from yawning_titan.notebooks.Agents.QBMagent import QBMAgent
from yawning_titan import _YT_ROOT_DIR
from matplotlib import pyplot as plt

class QBMResults:
    def __init__(self,Agent: QBMAgent,resultsFol):
        self.agent = Agent
        self.ResultsDir = os.path.join(_YT_ROOT_DIR,'results',resultsFol)
        if not os.path.isdir(self.ResultsDir):
            os.makedirs(self.ResultsDir)
            
    def toExcel(self):
        
        try:
            file = os.path.join(self.ResultsDir,'Log.xlsx')
            with pd.ExcelWriter(file, engine='openpyxl', mode='w+') as writer:  
                for key in list(self.agent.log.keys()):
                    df = pd.DataFrame(self.agent.log[key])
                    df.to_excel(writer, sheet_name=key)
        except:
            for key in list(self.agent.log.keys()):
                df = pd.DataFrame(self.agent.log[key])
                df.to_csv(os.path.join(self.ResultsDir,key+'.csv'))

    def plot(self,keys = [],showFigs = True,saveFigs = False,figSize=(7,4)):
        if keys == []:
            keys = list(self.agent.log.keys())
        for key in keys:
            df = pd.DataFrame(self.agent.log[key])
            isDiscrete = self.isDiscrete(key)


            fig = plt.figure(figsize=figSize)
            xdata = range(len(df))
            for iC,col in enumerate(df.columns):
                if isDiscrete:
                    plotI = df[col]==1
                    thisx = np.array(xdata)
                    thisx = thisx[plotI]
                    plt.scatter(thisx,iC*np.ones_like(thisx))
                else:
                    plt.plot(xdata,df[col],label=str(iC))
            plt.xlabel('Step')
            plt.ylabel(key)              
            if not isDiscrete:
                plt.legend()
            if saveFigs:
                plt.savefig(os.path.join(self.ResultsDir,key+'.png'))
            if showFigs:
                plt.show()
            else:
                plt.close(fig)

    def isDiscrete(self,key):
        df = pd.DataFrame(self.agent.log[key])
        return np.all(np.logical_or(df==0,df ==1))

