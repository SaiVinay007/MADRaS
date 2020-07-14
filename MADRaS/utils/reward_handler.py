import numpy as np
import warnings
import math
from copy import deepcopy


class RewardHandler(object):
    """Composes the reward function from a given reward configuration."""
    def __init__(self, cfg):
        self.rewards = {}
        for key in cfg:
            try:
                exec("self.rewards['{}'] = {}({})".format(key, key, cfg[key]))
            except:
                raise ValueError("Unknown reward class {}".format(key))

        if not self.rewards:
            warnings.warn("No reward function specified. Setting TorcsReward with "
                          "scale=1.0 as reward.")
            self.rewards['TorcsReward'] = TorcsReward({'scale': 1.0})

    def get_reward(self, game_config, game_state, action=None):
        reward = 0.0
        tmp = []
        for reward_function in self.rewards.values():
            # print(reward_function)
            val =  reward_function.compute_reward(game_config, game_state, action)
            reward += val
            tmp.append(val)
        # print(tmp)
        if math.isnan(reward):
            reward = 0.0
        # print(reward)
        return reward

    def reset(self):
        for reward_function in self.rewards.values():
            reward_function.reset()


class MadrasReward(object):
    """Base class of MADRaS reward function classes.
    Any new reward class must inherit this class and implement
    the following methods:
        - [required] compute_reward(game_config, game_state)
        - [optional] reset()
    """
    def __init__(self, cfg):
        self.cfg = cfg
        if "scale" not in self.cfg:
            self.cfg["scale"] = 1.0

    def compute_reward(self, game_config, game_state, action=None):
        del game_config, game_state
        raise NotImplementedError("Successor class must implement this method.")

    def reset(self):
        pass


class TorcsReward(MadrasReward):
    """Vanilla reward function provided by TORCS."""
    def compute_reward(self, game_config, game_state, action=None):
        del game_config
        if not math.isnan(game_state["torcs_reward"]):
            return self.cfg["scale"] * game_state["torcs_reward"]
        else:
            return 0.0


class ProgressReward(MadrasReward):
    """Proportional to the fraction of the track length traversed in one step."""
    def __init__(self, cfg):
        self.prev_dist = 0.0
        super(ProgressReward, self).__init__(cfg)

    def compute_reward(self, game_config, game_state, action=None):
        progress = game_state["distance_traversed"] - self.prev_dist
        reward = self.cfg["scale"] * (progress / game_config.track_len)
        self.prev_dist = deepcopy(game_state["distance_traversed"])
        return reward

    def reset(self):
        self.prev_dist = 0.0


class ProgressReward2(ProgressReward):
    def compute_reward(self, game_config, game_state, action=None):
        # target_speed = game_config.target_speed / 50  # m/step
        target_speed = game_config.target_speed / 10  # m/step
        progress = game_state["distance_traversed"] - self.prev_dist
        reward = self.cfg["scale"] * np.min([1.0, progress/target_speed])
        self.prev_dist = deepcopy(game_state["distance_traversed"])
        # print(" ProgressReward = ", reward, " progress = ", progress, " target_speed = ", target_speed, "current speed = ", progress/target_speed)
        return reward


class ProgressReward3(ProgressReward):
    def compute_reward(self, game_config, game_state, action=None):
        target_speed = game_config.target_speed / 50  # m/step
        progress = game_state["distance_traversed"] - self.prev_dist
        reward = self.cfg["scale"] * (progress/target_speed)
        self.prev_dist = deepcopy(game_state["distance_traversed"])
        return reward


class AvgSpeedReward(MadrasReward):
    def __init__(self, cfg):
        self.num_steps = 0
        super(AvgSpeedReward, self).__init__(cfg)

    def compute_reward(self, game_config, game_state, action=None):
        self.num_steps += 1
        if game_state["distance_traversed"] < game_config.track_len:
            return 0.0
        else:
            target_speed = game_config.target_speed / 50  # m/step
            target_num_steps = game_config.track_len / target_speed
            reward = self.cfg["scale"] * target_num_steps / self.num_steps
            return reward

class CollisionPenalty(MadrasReward):
    def __init__(self, cfg):
        self.damage = 0.0
        self.num_steps = 0
        super(CollisionPenalty, self).__init__(cfg)

    def compute_reward(self, game_config, game_state, action=None):
        del game_config
        reward = 0.0
        self.num_steps += 1
        if self.num_steps == 1:
            self.damage = game_state["damage"]

        if self.damage < game_state["damage"]:
            reward = -self.cfg["scale"]
        return reward


class TurnBackwardPenalty(MadrasReward):
    def compute_reward(self, game_config, game_state, action=None):
        del game_config
        reward = 0.0
        if np.cos(game_state["angle"]) < 0:
            reward = -self.cfg["scale"]
        return reward

class RankOneReward(MadrasReward):
    def compute_reward(self, game_config, game_state, action=None):
        if game_state["racePos"] == 1:
            reward = self.cfg["scale"]
        else:
            reward = 0.0
        return reward


class AngAcclPenalty(MadrasReward):
    def __init__(self, cfg):
        self.prev_angles = [0., 0.]
        self.threshold = cfg["max_ang_accl"]
        super(AngAcclPenalty, self).__init__(cfg)

    def reset(self):
        self.prev_angles = [0., 0.]

    def calc_ang_accl(self, angles):
        w1 = angles[1] - angles[0]
        w2 = angles[2] - angles[1]
        return w2 - w1

    def compute_reward(self, game_config, game_state, action=None):
        self.prev_angles.append(game_state["angle"])
        ang_accl = self.calc_ang_accl(self.prev_angles)
        if np.abs(ang_accl) > self.threshold:
            reward = - self.cfg["scale"] * np.abs(ang_accl) / self.threshold
        else:
            reward = 0
        self.prev_angles = self.prev_angles[1:]
        return reward


class SuccessfulOvertakeReward(MadrasReward):
    def __init__(self, cfg):
        self.rank = np.inf
        super(SuccessfulOvertakeReward, self).__init__(cfg)

    def reset(self):
        self.rank = np.inf

    def compute_reward(self, game_config, game_state, action=None):
        reward = 0.0
        if math.isinf(self.rank):  # very fist step
            self.rank = game_state["racePos"]
        elif game_state["racePos"] < self.rank:
            self.rank = game_state["racePos"]
            reward = self.cfg["scale"]
        return reward


class AccBrakePenality(MadrasReward):
    # If we apply both accelerator and brake at the same time for more than n steps : -ve reward
    # Helps to learn not to apply accelerator and brake at the same time
    def __init__(self, cfg):
        self.count = 0
        super(AccBrakePenality, self).__init__(cfg)
    
    def compute_reward(self, game_config, game_state, action=None):  
        # if action[1]>0 and action[2]>0:
        # print(action[1], action[2])

        if abs(action[1]-action[2])<0.6: # forcing to have a difference of atleast 0.6 
            self.count+=1
        
        if self.count >=self.cfg["count"]:
            reward = - self.cfg["scale"]
            self.count = 0
            return reward
        else:
            return 0


class StayStillPenality(MadrasReward):
    # If the acclerator is less than brake for more than 20 steps continously : -1 reward
    # Helps in favoring car to move than at rest
    def __init__(self, cfg):
        self.count = 0
        self.bool = False
        super(StayStillPenality, self).__init__(cfg)
    
    def compute_reward(self, game_config, game_state, action=None):

        if action[1]<action[2]:
            self.count += 1
        
        if self.count >= self.cfg["count"]:
            reward = - self.cfg["scale"]
            self.count = 0
            return reward
        else:
            return 0 


class ContSteerPenality(MadrasReward):
    # If car continously steers to one side for 50 steps : -10 reward
    # Helps in favoring car not to continously steer in one direction
    def __init__(self, cfg):
        self.count = 0
        self.prev_action = None
        super(ContSteerPenality, self).__init__(cfg)
    
    def compute_reward(self, game_config, game_state, action=None):
        
        if self.prev_action is not None :
            if action[0]*self.prev_action > 0 : # actions are of same sign
                if self.count == 0:
                    self.count += 2 # To account for the previous and current actions
                else:
                    self.count += 1
            else:
                self.count = 0
        self.prev_action = action[0]
        
        if self.count >= self.cfg["count"]:
            reward = - self.cfg["scale"]
            self.count = 0
            print("ContSteerPenality")
            return reward
        else:
            return 0 


class OutOfTrackPenality(MadrasReward):
    def __init__(self, cfg):
        super(OutOfTrackPenality, self).__init__(cfg)

    def compute_reward(self, game_config, game_state, action=None):
        self.track_limits = game_config.track_limits
        if (game_state["trackPos"] < self.track_limits['low'] or 
            game_state["trackPos"] > self.track_limits['high'] or 
            np.any(np.asarray(game_state["track"]) < 0)):
            reward = -self.cfg["scale"] 
            print("out of track penality", reward)
            return reward
        else:
            return 0


class LessProgressPenality(MadrasReward):
    def __init__(self, cfg):
        self.num_steps = 0
        super(LessProgressPenality, self).__init__(cfg)

    def compute_reward(self, game_config, game_state, action=None):
        self.num_steps += 1
        progress = game_state["distance_traversed"]
        if self.num_steps >= self.cfg["steps"] and progress<self.cfg["progress"]:
            self.num_steps = 0
            reward = - self.cfg["scale"]
            print("LessProgressPenality", reward)
            return reward
        else:
            return 0