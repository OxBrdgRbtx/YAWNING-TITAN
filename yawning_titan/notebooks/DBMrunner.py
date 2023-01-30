from stable_baselines3.common.env_checker import check_env

from yawning_titan.envs.generic.core.blue_interface import BlueInterface
from yawning_titan.envs.generic.core.red_interface import RedInterface
from yawning_titan.envs.generic.generic_env import GenericNetworkEnv
from yawning_titan.envs.generic.helpers import network_creator
from yawning_titan.envs.generic.core.action_loops import ActionLoop
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.config.network_config.network_config import NetworkConfig
from yawning_titan.config.game_modes import default_game_mode_path
from yawning_titan.config.game_config.game_mode_config import GameModeConfig
import numpy as np

# Define network
#matrix, node_positions = network_creator.create_18_node_network() 
# Simple Network of two nodes:
matrix = np.asarray(
        [
            [0, 1],
            [1, 0],
        ]
    )
nodePositions = {
        "0": [1, 0],
        "1": [2, 0],
    }
entryNodes = ['0']
highValueNodes = ['1']
    
network_config = NetworkConfig.create_from_args(matrix=matrix, positions=nodePositions,entry_nodes=entryNodes,high_value_nodes=highValueNodes)


import os
import pathlib
from yawning_titan.config import _LIB_CONFIG_ROOT_PATH

path = pathlib.Path(
        os.path.join(
            _LIB_CONFIG_ROOT_PATH,
            "_package_data",
            "game_modes",
            "default_game_mode_basic.yaml",
        )
    )

# game_mode_config = GameModeConfig.create_from_yaml(default_game_mode_path())
game_mode_config = GameModeConfig.create_from_yaml(path)

network_interface = NetworkInterface(game_mode=game_mode_config, network=network_config)

# Define agents
red = RedInterface(network_interface)
blue = BlueInterface(network_interface)


env = GenericNetworkEnv(red, blue, network_interface)

check_env(env, warn=True)

_ = env.reset()

from yawning_titan.notebooks.Agents.QBMagent import QBMAgent
from yawning_titan.notebooks.QBMResults import QBMResults

agent = QBMAgent(env)
# agent.initRBM(3) # 3 hidden nodes
agent.initDBM([3,3]) # 3 hidden nodes in each layer

log = agent.learn(nSteps=5000)

results = QBMResults(agent,'DBMtest_Simulate')
results.toExcel()
results.plot(showFigs=False,saveFigs=True)