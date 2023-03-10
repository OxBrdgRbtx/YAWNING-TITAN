import os
import pathlib

def basicGameRules():
    path = pathlib.Path(
            os.path.join(
                "game_modes",
                "default_game_mode_basic.yaml",
            )
        )
    return path

def QBMGameRules():
    path = pathlib.Path(
            os.path.join(
                "game_modes",
                "QBM_game_mode.yaml",
            )
        )
    return path

def QBMGameRules_extraObservation():
    path = pathlib.Path(
            os.path.join(
                "game_modes",
                "QBM_game_mode_extendedObservation.yaml",
            )
        )
    return path

def QBMGameRules_extraAction():
    path = pathlib.Path(
            os.path.join(
                "game_modes",
                "QBM_game_mode_extendedAction.yaml",
            )
        )
    return path

def QBMGameRules_extraActionObservation():
    path = pathlib.Path(
            os.path.join(
                "game_modes",
                "QBM_game_mode_extended.yaml",
            )
        )
    return path

def defaultYTRules():
    return None
