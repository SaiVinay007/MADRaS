"""
Gym Madras Env Wrapper.

This is an OpenAI gym environment wrapper for the MADRaS simulator. For more information on the OpenAI Gym interface, please refer to: https://gym.openai.com

Built on top of gym_torcs https://github.com/ugo-nama-kun/gym_torcs/blob/master/gym_torcs.py

The following enhancements were made for Multi-agent synchronization using exception handling:
- All the agents connect to the same TORCS engine through UDP ports
- If an agent fails to connect to the TORCS engine, it keeps on trying in a loop until successful
- Restart the episode for all the agents when any one of the learning agents terminates its episode

"""

import math
from copy import deepcopy
import numpy as np
import MADRaS.utils.snakeoil3_gym as snakeoil3
from MADRaS.utils.gym_torcs import TorcsEnv
from MADRaS.controllers.pid import PIDController
import gym
from gym.utils import seeding
import os
import subprocess
import signal
import time
from mpi4py import MPI
import random
import socket
import yaml
import MADRaS.utils.reward_manager as rm
import MADRaS.utils.done_manager as dm
import MADRaS.utils.observation_manager as om
import MADRaS.utils.torcs_server_config as torcs_config
import MADRaS.traffic.traffic as traffic
import logging
logger = logging.getLogger(__name__)

from MADRaS.envs.gym_madras import MadrasEnv, MadrasConfig

path_and_file = os.path.realpath(__file__)
path, file = os.path.split(path_and_file)
DEFAULT_SIM_OPTIONS_FILE = os.path.join(path, "data", "sim_options.yml")

cfg_path = '/home/saivinay/Documents/RL_project/MADRaS/MADRaS/envs/data/sim_options.yml'

def parse_yaml(yaml_file):
    if not yaml_file:
        yaml_file = DEFAULT_SIM_OPTIONS_FILE
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


env = MadrasEnv()

_config = MadrasConfig()
_config.update(parse_yaml(cfg_path))

# print(_config.server_config)
torcs_server_config = torcs_config.TorcsConfig(_config.server_config, randomize=_config.randomize_env)
torcs_server_config.generate_torcs_server_config()

traffic_manager = traffic.MadrasTrafficManager(_config.torcs_server_port, 1, _config.traffic)
# traffic_manager.flag_off_traffic()
# traffic_manager.kill_all_traffic_agents()


traffic_manager.reset(torcs_server_config.num_traffic_cars)
