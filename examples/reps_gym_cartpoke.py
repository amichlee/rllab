from __future__ import print_function

from rllab.algos.reps import REPS
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
from rllab.policies.categorical_mlp_policy import CategoricalMLPPolicy
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.box2d.mountain_car_env import MountainCarEnv

import sys
import os
# import gc, ppri nt  

_gym_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../OpSpaceLearning/python/"))
_csaienv_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../OpSpaceLearning/bin/"))

sys.path.append(_gym_dir)
sys.path.append(_csaienv_dir)
os.chdir(_gym_dir)


import gym

import gym_sai2


# print(gym.envs.registry.all())
# env = normalize(GymEnv("cy-saienv-peg1-v0",  force_reset=True))
# # env.init()


# policy = GaussianMLPPolicy(
#     env_spec=env.spec,
#     hidden_sizes=(32, 32)
# )
# baseline = LinearFeatureBaseline(env_spec=env.spec)
# algo = REPS(
#     env=env,
#     policy=policy,
#     baseline=baseline,
#     # n_samples = 1,
#     batch_size = 800,
#     max_path_length=80,
#     n_itr=500,
#     init_std=0,
#     extra_decay_time=50,
#     extra_std=0.1,
# )
# algo.train()





def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    env = normalize(MountainCarEnv())

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(32, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)
    algo = REPS(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=2000,
        max_path_length=env.horizon,
        n_itr=120,
        # discount=0.99,
        # step_size=0.01,
        # Uncomment both lines (this and the plot parameter below) to enable plotting
        # plot=True,
# 


    )
    algo.train()



run_experiment_lite(
    run_task,
    # # Number of parallel workers for sampling
    # n_parallel=1,
    # # Only keep the snapshot parameters for the last iteration
    # snapshot_mode="last",
    # # Specifies the seed for the experiment. If this is not provided, a random seed
    # # will be used
    # seed=1,
    # plot=True,
)
