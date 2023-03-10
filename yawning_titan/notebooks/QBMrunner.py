# Get Env checker
from stable_baselines3.common.env_checker import check_env

# Get YT interfaces
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.config.game_modes import default_game_mode_path

# Get agent and definition functions
from yawning_titan.notebooks.Agents.QBMagentBatch import QBMBatchAgent
from yawning_titan.notebooks.Agents.QBMagent import QBMAgent
from yawning_titan.notebooks.gameDefinitions.networkDefinitions import *
from yawning_titan.notebooks.gameDefinitions.gameModes import *


#%% Define game rules and network
game_mode_config = QBMGameRules()
network_config = get10NodeNetwork()

#%% Build network interface with games rules
network_interface = NetworkInterface(game_mode=game_mode_config, network=network_config)

#%% Define agents and environment
red = RedInterface(network_interface)
blue = BlueInterface(network_interface)
env = GenericNetworkEnv(red, blue, network_interface)

#%% Check and reset environment
check_env(env, warn=True)
_ = env.reset()

#%% Set scenario baselines
nSteps = int(2.5e5)
printRate = 10000 # nSteps between printing to log/terminal
dbmSize = [16,16] # DBM shape, RBM is set up to have equal number of hidden nodes
epsilon = 0.1
gamma = 0.95
pRandomDecay = 0.99
minPrandom = 0.05
name = '25e4_32Units'
longName = '1e6_32Units'
longRBMscale = 4

# %% Build an RBM agent and learn via explicit calculation
agent = QBMAgent(env,'FullRuns\\RBM_'+name,
                 epsilon=epsilon,gamma=gamma,printRate=printRate,
                 pRandomDecay=pRandomDecay,minPrandom=minPrandom,
                 writeWeights=True)
agent.initRBM(sum(dbmSize))
agent.learn(nSteps=nSteps)
agent.exportResults()

agent = QBMAgent(env,'FullRuns\\RBM_'+longName,
                 epsilon=epsilon,gamma=gamma,printRate=printRate*longRBMscale,
                 pRandomDecay=pRandomDecay,minPrandom=minPrandom,
                 writeWeights=True)
agent.initRBM(sum(dbmSize))
agent.learn(nSteps=nSteps*longRBMscale)
agent.exportResults()


# # #%% Build a DBM agent and learn via batched Quantum Annealing
# agent = QBMBatchAgent(env,'FullRuns\\QBM_'+name+'QuantumBatch',
#               epsilon=epsilon,gamma=gamma,printRate=printRate,
#               pRandomDecay=pRandomDecay,minPrandom=minPrandom,SimulateAnneal=False,
#               SimulateAnnealForAction=False,AnnealToBestAction=True,writeWeights=True)
# agent.setBatchSize(batchSize=1)
# agent.initDBM(dbmSize)
# agent.learn(nSteps=nSteps)
# agent.exportResults()

