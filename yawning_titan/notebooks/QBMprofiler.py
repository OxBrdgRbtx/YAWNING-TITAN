# Get Env checker
from stable_baselines3.common.env_checker import check_env

# Get YT interfaces
from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.generic_env_BlueFirst import GenericNetworkEnvBF
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
nSteps = int(100)
printRate = 20 # nSteps between printing to log/terminal
# dbmSize = [16,16] # DBM shape, RBM is set up to have equal number of hidden nodes
# dbmSize = [8,8] # DBM shape, RBM is set up to have equal number of hidden nodes
dbmSize = [4,4] # DBM shape, RBM is set up to have equal number of hidden nodes
epsilon = 0.1
gamma = 0.95
pRandomDecay = 0.99 # Added for 1000 step case - Smaller pRandom than default so decay happens within 1000 steps
minPrandom = 0.05
# name = '_32Nodes_profile_3_'
# name = '_16Nodes_profile_3_'
name = '_8Nodes_profile_3_'

#%% Build an RBM agent and learn via explicit calculation
agent = QBMAgent(env,'QBM'+name+'RBM',
                 epsilon=epsilon,gamma=gamma,printRate=printRate,
                 pRandomDecay=pRandomDecay,minPrandom=minPrandom,
                 writeWeights=True)
                #  SimulateAnnealForAction=True,AnnealToBestAction=True,writeWeights=True)
agent.initRBM(sum(dbmSize))
agent.learn(nSteps=nSteps*2)
agent.exportResults(saveFigs=False)

# # #%% Build a DBM agent and learn via Simulated Annealing
# agent = QBMAgent(env,'QBM'+name+'Simulated',
#               epsilon=epsilon,gamma=gamma,printRate=printRate,
#               pRandomDecay=pRandomDecay,minPrandom=minPrandom)
# agent.initDBM(dbmSize)
# agent.learn(nSteps=nSteps)
# agent.exportResults(saveFigs=False)

# #%% Build a DBM agent and learn via Simulated Annealing
agent = QBMBatchAgent(env,'QBM'+name+'SimulatedBatch',
              epsilon=epsilon,gamma=gamma,printRate=printRate,
              pRandomDecay=pRandomDecay,minPrandom=minPrandom,
              SimulateAnnealForAction=True,AnnealToBestAction=True)
agent.setBatchSize(batchSize=1)
agent.initDBM(dbmSize)
agent.learn(nSteps=nSteps)
agent.exportResults(saveFigs=False)

# #%% Build a DBM agent and learn via Quantum Annealing
# agent = QBMAgent(env,'QBM'+name+'Quantum',
#               epsilon=epsilon,gamma=gamma,printRate=printRate,
#               pRandomDecay=pRandomDecay,minPrandom=minPrandom,SimulateAnneal=False,
#               SimulateAnnealForAction=True,AnnealToBestAction=True)
# agent.initDBM(dbmSize)
# agent.learn(nSteps=nSteps)
# agent.exportResults(saveFigs=False)


# agent = QBMBatchAgent(env,'QBM'+name+'Quantum_1Batch',
#               epsilon=epsilon,gamma=gamma,printRate=printRate,
#               pRandomDecay=pRandomDecay,minPrandom=minPrandom,SimulateAnneal=False,
#               SimulateAnnealForAction=True,AnnealToBestAction=True)
# agent.setBatchSize(batchSize=1)
# agent.initDBM(dbmSize)
# agent.learn(nSteps=nSteps)
# agent.exportResults(saveFigs=False)


# # #%% Build a DBM agent and learn via batched Quantum Annealing
# agent = QBMBatchAgent(env,'QBM'+name+'Quantum_5Batch',
#               epsilon=epsilon,gamma=gamma,printRate=printRate,
#               pRandomDecay=pRandomDecay,minPrandom=minPrandom,SimulateAnneal=False,
#               SimulateAnnealForAction=True,AnnealToBestAction=True)
# agent.setBatchSize(batchSize=5)
# agent.initDBM(dbmSize)
# agent.learn(nSteps=nSteps)
# agent.exportResults(saveFigs=False)

#%% Build a DBM agent and learn via batched Quantum Annealing
agent = QBMBatchAgent(env,'QBM'+name+'Quantum_1Batch_Qaction',
              epsilon=epsilon,gamma=gamma,printRate=printRate,
              pRandomDecay=pRandomDecay,minPrandom=minPrandom,SimulateAnneal=False,
              SimulateAnnealForAction=False,AnnealToBestAction=True)
agent.setBatchSize(batchSize=1)
agent.initDBM(dbmSize)
agent.learn(nSteps=nSteps)
agent.exportResults(saveFigs=False)
