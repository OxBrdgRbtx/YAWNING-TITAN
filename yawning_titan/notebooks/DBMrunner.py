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
from yawning_titan.notebooks.Agents.QBMagent import QBMAgent
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
agent = QBMAgent(env,'RBMtest')
agent.initRBM(8) # 8 hidden nodes
agent.learn(nSteps=1e6)
agent.exportResults()


#%% Build a DBM agent and learn
agent = QBMAgent(env,'DBMtest')
agent.initDBM([4,4]) # 8 hidden nodes
agent.learn(nSteps=5e5)
agent.exportResults()
