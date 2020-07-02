import ray
import ray.rllib.agents.ppo as ppo
# import ray.rllib.agents.pg as pg
from ray.tune.logger import pretty_print
import rllib_helpers as helpers

import logging.config
import sys
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
import glob
import os
import traceback
helpers.register_madras()
# ray.init()
ray.init(lru_evict=True)
config = ppo.DEFAULT_CONFIG.copy()
# config = pg.DEFAULT_CONFIG.copy()

# Full config is here: https://github.com/ray-project/ray/blob/d51583dbd6dc9c082764b9ec06349678aaa71078/rllib/agents/trainer.py#L42
config["num_gpus"] = 1
config["num_workers"] = 1  # 12 works
config["eager"] = False
config["vf_clip_param"] = 20  # originally it was 10. We should consider scaling down the rewards for keeping episode reward under 2000
# ------
# config["gamma"] = 0.7
# config["lr"] = 5e-7
# config["batch_mode"] = "complete_episodes"
# config["train_batch_size"] = 10000

trainer = ppo.PPOTrainer(config=config, env="madras_env")
# trainer = pg.PGTrainer(config=config, env="madras_env")


# Can optionally call trainer.restore(path) to load a checkpoint.
checkpoint_dir = '/home/saivinay/ray_results/PPO_madras_env_2020-05-20_12-48-02ghj0ji1k/'
# temp = os.listdir(checkpoint_dir)
# temp = glob.glob(checkpoint_dir+'/checkpoint*')
path = checkpoint_dir+'checkpoint_121/checkpoint-121'

# To continue from checkpoint
# if os.path.exists(path):
#     print('Restored')
#     trainer.restore(path)

# policy = trainer.get_policy()
# print(policy.model.base_model.summary())

iterations = 1000

for i in range(iterations):
   # Perform one iteration of training the policy with PPO
   result = trainer.train()
   print(pretty_print(result))
   exit()
   if i % 10 == 0:
       checkpoint = trainer.save()
       logging.info("checkpoint saved at", checkpoint)
