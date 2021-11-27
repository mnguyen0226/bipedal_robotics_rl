# Implementation of Bipedal Walking with A2C - Advantage Actor Critics
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

import gym
import os
import sys
import time 

from utils import *
# from models.mlp_policy import Policy
# from models.mlp_critic import Value
from models.policy import Policy
from models.critic import Value
from models.mlp_policy_disc import DiscretePolicy
from core.a2c import a2c_step
from core.common import estimate_advantages
from core.agent import Agent