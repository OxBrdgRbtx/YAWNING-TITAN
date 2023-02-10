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
from yawning_titan.notebooks.gameDefinitions.networkDefinitions import *
from yawning_titan.notebooks.gameDefinitions.gameModes import *


#%% Define game rules and network
game_mode_config = basicGameRules()
network_config = getFiveNodeNetwork()

#%% Build network interface with games rules
network_interface = NetworkInterface(game_mode=game_mode_config, network=network_config)

#%% Define agents and environment
red = RedInterface(network_interface)
blue = BlueInterface(network_interface)
env = GenericNetworkEnvBF(red, blue, network_interface)

#%% Check and reset environment
check_env(env, warn=True)
_ = env.reset()

#%% Build an RBM agent and learn
agent = QBMBatchAgent(env,'RBMBatchtest_1')#,beta=5,epsilon=1e-3,adaptiveGradient=False)
agent.initRBM(8) # 8 hidden nodes
agent.setBatchSize(batchSize=1)
agent.learn(nSteps=3e5)
agent.exportResults()

agent = QBMBatchAgent(env,'RBMBatchtest_2')#,beta=5,epsilon=1e-3,adaptiveGradient=False)
agent.initRBM(8) # 8 hidden nodes
agent.setBatchSize(batchSize=2)
agent.learn(nSteps=3e5)
agent.exportResults()

agent = QBMBatchAgent(env,'RBMBatchtest_5')#,beta=5,epsilon=1e-3,adaptiveGradient=False)
agent.initRBM(8) # 8 hidden nodes
agent.setBatchSize(batchSize=5)
agent.learn(nSteps=3e5)
agent.exportResults(writeTables=False)

agent = QBMBatchAgent(env,'RBMBatchtest_10')#,beta=5,epsilon=1e-3,adaptiveGradient=False)
agent.initRBM(8) # 8 hidden nodes
agent.setBatchSize(batchSize=10)
agent.learn(nSteps=3e5) 
agent.exportResults(writeTables=False)

agent = QBMBatchAgent(env,'RBMBatchtest_30')#,beta=5,epsilon=1e-3,adaptiveGradient=False)
agent.initRBM(8) # 8 hidden nodes
agent.setBatchSize(batchSize=30)
agent.learn(nSteps=3e5)
agent.exportResults(writeTables=False)

