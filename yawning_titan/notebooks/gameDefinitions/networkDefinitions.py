import numpy as np
from yawning_titan.envs.generic.core.network_interface import NetworkInterface
from yawning_titan.envs.generic.helpers import network_creator

def getTwoNodeNetwork(settings):
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
    highValueNodes = '1'
    
    network_config = NetworkInterface(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_target=highValueNodes,
        settings_path=settings)
    return network_config

def getFiveNodeNetwork(settings):
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
    highValueNodes = '4'
    
    network_config = NetworkInterface(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_target=highValueNodes,
        settings_path=settings)
    return network_config

def get10NodeNetwork(settings):
    matrix, nodePositions = network_creator.create_18_node_network() 

    keep = [1, 3, 5, 7, 8, 9, 11, 12, 14, 16]
    matrix = matrix[keep,:]
    matrix = matrix[:,keep]
    nodePositions = {str(iN):nodePositions[str(iK)] for iN, iK in enumerate(keep)}

    entryNodes = ['2', '3', '7']
    highValueNodes = '5'

    network_config = NetworkInterface(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_target=highValueNodes,
        settings_path=settings)
    return network_config   

def get18NodeNetwork(settings):
    matrix, nodePositions = network_creator.create_18_node_network() 

    entryNodes = ['12', '5']
    highValueNodes = '10'

    network_config = NetworkInterface(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_target=highValueNodes,
        settings_path=settings)
    return network_config   


def get24NodeNetwork(settings):
    matrix_, nodePositions = network_creator.create_18_node_network() 

    nodePositions['18'] = [1,5]
    nodePositions['19'] = [2,5]
    nodePositions['20'] = [4,5]
    nodePositions['21'] = [1,3]
    nodePositions['22'] = [2,3] 
    nodePositions['23'] = [4,3]
    
    matrix = np.zeros((24,24))
    matrix[0:18,0:18] = matrix_
    for ii in [6, 19]:
        matrix[18,ii] = 1
        matrix[ii,18] = 1
    for ii in [18, 5, 7]:
        matrix[19,ii] = 1
        matrix[ii,19] = 1
    for ii in [5, 7, 8, 9]:
        matrix[20,ii] = 1
        matrix[ii,20] = 1    
    for ii in [6,22]:
        matrix[21,ii] = 1
        matrix[ii,21] = 1
    for ii in [21,12,7]:
        matrix[22,ii] = 1
        matrix[ii,22] = 1
    for ii in [12,7,8,11]:
        matrix[23,ii] = 1
        matrix[ii,23] = 1        



    entryNodes = ['12', '5']
    highValueNodes = '10'



    network_config = NetworkInterface(matrix=matrix,
        positions=nodePositions,
        entry_nodes=entryNodes,
        high_value_target=highValueNodes,
        settings_path=settings)
    return network_config   
