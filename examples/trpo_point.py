from rllab.algos.cem import CEM
from rllab.baselines.linear_feature_baseline import LinearFeatureBaseline
from point_env import PointEnv
from rllab.envs.normalized_env import normalize
from rllab.policies.gaussian_mlp_policy import GaussianMLPPolicy
from rllab.envs.base import Env
from rllab.spaces import Box
from rllab.envs.base import Step
import numpy as np


# class PointEnv(Env):
#     @property
#     def observation_space(self):
#         return Box(low=-np.inf, high=np.inf, shape=(2,))

#     @property
#     def action_space(self):
#         return Box(low=-0.1, high=0.1, shape=(2,))

#     def reset(self):
#         self._state = np.random.uniform(-1, 1, size=(2,))
#         observation = np.copy(self._state)
#         return observation

#     def step(self, action):
#         self._state = self._state + action
#         x, y = self._state
#         reward = - (x ** 2 + y ** 2) ** 0.5
#         done = abs(x) < 0.01 and abs(y) < 0.01
#         next_observation = np.copy(self._state)
#         return Step(observation=next_observation, reward=reward, done=done)

#     def render(self):
#         print('current state:', self._state)


env = normalize(PointEnv())

print(env.spec)
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = CEM(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
