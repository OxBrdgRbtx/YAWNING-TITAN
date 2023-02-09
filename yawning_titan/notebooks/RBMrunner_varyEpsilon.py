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

epsilonPowers = [-1,-2,-3,-4]
for epsilonPower in epsilonPowers:
    game_mode_config = QBMGameRules()
    name = 'RBM_basic_16_E'+str(epsilonPower)
    network_config = get10NodeNetwork()

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
    agent = QBMAgent(env,name,epsilon=10**epsilonPower)
    agent.initRBM(16)
    agent.learn(nSteps=5e5)
    agent.exportResults(writeTables=False)
