# Implementation of Bipedal Walking with PPO - Proximal Policy Optimization
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

import gym
import torch
import os
import sys
import pickle
import time
import math
from os import path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.utils.additional_torch import to_device
from algorithms.ppo_algorithm import Policy
from algorithms.ppo_algorithm import Value
from algorithms.ppo_algorithm import ppo_step
from algorithms.a2c_algorithm import estimate_advantages
from algorithms.a2c_algorithm import Agent
from algorithms.utils import ZFilter
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

################################
l2_reg = 1e-3
gamma = 0.99
tau = 0.95
max_num_iter = 1000  # 50000
render = False
min_batch_size = 2048
log_interval = 1
save_model_interval = 2
clip_epsilon = 0.2
env_name = "BipedalWalker-v2"
optim_epochs = 10
optim_batch_size = 64

###############################
dtype = torch.float64
torch.set_default_dtype(dtype)

# set device
device = (
    torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# environment
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)

# seeding
np.random.seed(1)
torch.manual_seed(1)
env.seed(1)

# define actor and critic network
policy_net = Policy(state_dim, env.action_space.shape[0], log_std=-1.0)
value_net = Value(state_dim)
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=4e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=8e-4)

# create agent
agent = Agent(
    env,
    policy_net,
    device,
    running_state=running_state,
    render=False,
    num_threads=1,
)


def assets_dir():
    return path.abspath(path.join(path.dirname(path.abspath(__file__)), "../assets"))


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)
        fixed_log_probs = policy_net.get_log_prob(states, actions)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(
        rewards, masks, values, gamma, tau, device
    )

    """perform mini batch PPO update"""
    optim_iter_num = int(math.ceil(states.shape[0] / optim_batch_size))
    for _ in range(optim_epochs):
        perm = np.arange(states.shape[0])
        np.random.shuffle(perm)
        perm = torch.LongTensor(perm).to(device)

        states, actions, returns, advantages, fixed_log_probs = (
            states[perm].clone(),
            actions[perm].clone(),
            returns[perm].clone(),
            advantages[perm].clone(),
            fixed_log_probs[perm].clone(),
        )

        for i in range(optim_iter_num):
            ind = slice(
                i * optim_batch_size, min((i + 1) * optim_batch_size, states.shape[0])
            )
            states_b, actions_b, advantages_b, returns_b, fixed_log_probs_b = (
                states[ind],
                actions[ind],
                advantages[ind],
                returns[ind],
                fixed_log_probs[ind],
            )

            ppo_step(
                policy_net,
                value_net,
                optimizer_policy,
                optimizer_value,
                1,
                states_b,
                actions_b,
                returns_b,
                advantages_b,
                fixed_log_probs_b,
                clip_epsilon,
                l2_reg,
            )


###############################
def ppo_main():
    # plot
    plot = plt.figure()
    xval, yval = [], []
    subplot = plot.add_subplot()
    plt.xlabel("Number Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Number Episodes")
    (plotLine,) = subplot.plot(xval, yval)
    subplot.set_xlim([0, max_num_iter])
    subplot.set_ylim([-400, 400])

    # run iteration
    for i_iter in range(max_num_iter):
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(min_batch_size, render)

        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % log_interval == 0:
            print(
                "{}\tT_sample {:.4f}\tT_update {:.4f}\tR_min {:.2f}\tR_max {:.2f}\tR_avg {:.2f}".format(
                    i_iter,
                    log["sample_time"],
                    t1 - t0,
                    log["min_reward"],
                    log["max_reward"],
                    log["avg_reward"],
                )
            )

        # plot
        xval.append(i_iter)
        yval.append(log["max_reward"])
        plotLine.set_xdata(xval)
        plotLine.set_ydata(yval)
        plot.savefig("./results/ppo_max_reward")

        if save_model_interval > 0 and (i_iter + 1) % save_model_interval == 0:
            to_device(torch.device("cpu"), policy_net, value_net)

            pickle.dump(
                (policy_net, value_net, running_state),
                open(
                    os.path.join(
                        assets_dir(),
                        "learned_models/ppo_algorithm/bipedal_walker_v2_ppo.p",
                    ),
                    "wb",
                ),
            )
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()

    print("All episodes finished.")


if __name__ == "__main__":
    ppo_main()
