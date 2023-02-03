import numpy as np
from yawning_titan.config.network_config.network_config import NetworkConfig
from yawning_titan.envs.generic.helpers import network_creator

def getTwoNodeNetwork():
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
    
    network_config = NetworkConfig.create_from_args(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_nodes=highValueNodes)
    return network_config

def getFiveNodeNetwork():
    # Simple Network of five nodes:
    matrix = np.asarray(
            [
                [0, 1, 1, 0, 0],
                [1, 0, 0, 1, 0],
                [1, 0, 0, 1, 0],
                [0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0]
            ]
        )
    nodePositions = {
            "0": [1, 0],
            "1": [2, 1],
            "2": [2,-1],
            "3": [3, 0],
            "4": [4, 0],
        }
    entryNodes = ['0']
    highValueNodes = ['4']
    
    network_config = NetworkConfig.create_from_args(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_nodes=highValueNodes)
    return network_config


def get18NodeNetwork():
    matrix, nodePositions = network_creator.create_18_node_network() 

    network_config = NetworkConfig.create_from_args(matrix=matrix,
        positions=nodePositions)
    return network_config    