import numpy as np
import gym
from envs.gym_madras_v2 import MadrasEnv
import os


def test_madras_vanilla():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(20000):
        obs, r, done, _ = env.step([[0.0, 1.0, -1.0]])
        print("{}: reward={}, done={}".format(t, r, done))
        dones = [x for x in done.values()]
        if np.all(dones):
            env.reset()
    os.system("pkill torcs")


def test_madras_pid():
    env = MadrasEnv()
    print("Testing reset...")
    obs = env.reset()
    print("Initial observation: {}."
          " Verify if the number of dimensions {} is right.".format(obs, len(obs)))
    print("Testing step...")
    for t in range(2000):
        obs, r, done, _ = env.step([[0.3, 0.5],
                                    # [0.0, 1.0]
                                    ])
        print("{}: reward={}, done={}".format(t, r, done))
        dones = [x for x in done.values()]
        if np.all(dones):
            env.reset()
    os.system("pkill torcs")


if __name__=='__main__':
    # test_madras_vanilla()
    test_madras_pid()