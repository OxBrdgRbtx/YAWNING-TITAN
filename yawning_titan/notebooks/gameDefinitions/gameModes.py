import os
import pathlib
from yawning_titan.config import _LIB_CONFIG_ROOT_PATH
from yawning_titan.config.game_config.game_mode_config import GameModeConfig
from yawning_titan.config.game_modes import default_game_mode_path

def basicGameRules():
    path = pathlib.Path(
            os.path.join(
                _LIB_CONFIG_ROOT_PATH,
                "_package_data",
                "game_modes",
                "default_game_mode_basic.yaml",
            )
        )
    return GameModeConfig.create_from_yaml(path)

def defaultYTRules():
    return GameModeConfig.create_from_yaml(default_game_mode_path())
