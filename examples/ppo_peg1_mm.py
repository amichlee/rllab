from rllab.algos.ppo import PPO
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from rllab.envs.box2d.cartpole_env import CartpoleEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy

from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import run_experiment_lite
import sys
import os
# import gc, ppri nt  



import gym

from gym_sai2.envs.peg1_multimodal_env import Peg1MultimodalEnv




def run_task(*_):
    # Please note that different environments with different action spaces may require different
    # policies. For example with a Box action space, a GaussianMLPPolicy works, but for a Discrete
    # action space may need to use a CategoricalMLPPolicy (see the trpo_gym_cartpole.py example)
    
    env = normalize(GymEnv("peg1-multimodal-v0",  force_reset=True))
        # env.init()
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        # The neural network policy should have two hidden layers, each with 32 hidden units.
        hidden_sizes=(128, 64, 32)
    )

    baseline = LinearFeatureBaseline(env_spec=env.spec)

    algo = PPO(
        env=env,
        policy=policy,
        baseline=baseline,
        batch_size=100,
        max_path_length=10,
        n_itr=201,
        discount=0.99,
        step_size=0.01,
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
