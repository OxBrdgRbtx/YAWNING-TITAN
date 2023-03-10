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

def get10NodeNetwork():
    matrix, nodePositions = network_creator.create_18_node_network() 

    keep = [1, 3, 5, 7, 8, 9, 11, 12, 14, 16]
    matrix = matrix[keep,:]
    matrix = matrix[:,keep]
    nodePositions = {str(iN):nodePositions[str(iK)] for iN, iK in enumerate(keep)}

    entryNodes = ['2', '3', '7']
    highValueNodes = ['5']

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