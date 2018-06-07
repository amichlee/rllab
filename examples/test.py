from __future__ import print_function
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

env = gym.make("cy-saienv-peg1-v0")