# Implementation of Bipedal Walking with A2C + GAE - Advantage Actor Critics with Generalized Advantage Estimation
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

# Inspired Reference: https://github.com/lnpalmer/A2C
# Inspired Reference: https://github.com/floodsung/a2c_cartpole_pytorch
# Reading: https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html


import gym
import torch
import os
import sys
import pickle
import time
from os import path
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from algorithms.utils.additional_torch import to_device
from algorithms.a2c_algorithm import Policy
from algorithms.a2c_algorithm import Value
from algorithms.a2c_algorithm import a2c_step
from algorithms.a2c_algorithm import generalized_advantage_estimation
from algorithms.a2c_algorithm import BipedalWalkerAgent
from algorithms.utils import AlgorithmRunManagement
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# initialize global variable
# L2 regularization can deal with the multicollinearity (independent variables are highly correlated)
# problems through constricting the coefficient and by keeping all the variables.
L2_REG = 1e-3 # calculate error with L2 regularization
GAMMA = 0.99  # discount factor
MAX_NUM_ITER = 1000  # 5000
RENDER = True  # True
MIN_BATCH_SIZE = 2048
LOG_INTERVAL = 1  # print out rate
SAVE_MODEL_INTERVAL = 100
ENV_NAME = "BipedalWalker-v2"

# set datatype
dtype = torch.float64
torch.set_default_dtype(dtype)

# set device
device = (
    torch.device("cuda", index=0) if torch.cuda.is_available() else torch.device("cpu")
)
if torch.cuda.is_available():
    torch.cuda.set_device(0)

# environment
env = gym.make(ENV_NAME)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = AlgorithmRunManagement((state_dim,), clip=5)

# seeding
np.random.seed(1)
torch.manual_seed(1)
env.seed(1)

# define actor and critic network
policy_net = Policy(state_dim, env.action_space.shape[0], log_std=-1.0)
value_net = Value(state_dim)

# Test Trained Loaded
# policy_net, value_net, running_state = pickle.load(
#     open("assets/learned_models/a2c_algorithm/cpu_bipedal_walker_v2_a2c.p", "rb")
# )

policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=4e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=8e-4)

# create Bipedal Walker Agent
agent = BipedalWalkerAgent(
    env,
    policy_net,
    device,
    running_state=running_state,
    render=False,
    num_threads=1,
)


def saved_assets_dir():
    """Saves trained model in assets directory

    Returns:
        Paths to asset directory
    """
    return path.abspath(
        path.join(path.dirname(path.abspath(__file__)), "../../../assets")
    )


def update_a2c_params(batch, tau):
    """Updates training parameters by taking steps from A2C algorithm

    Args:
        batch: input batch
    """
    # collect all state, action, reward, next state by stacking batch. 
    # thus this is similar to "for each step in the episode do" when run
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    (
        advantages,
        returns,
    ) = generalized_advantage_estimation(  # get estimated advantage from stepping trajectories
        rewards=rewards, masks=masks, values=values, gamma=GAMMA, tau=tau, device=device
    )

    a2c_step(  # run A2C algorithm updates
        policy_net=policy_net,
        value_net=value_net,
        optimizer_policy=optimizer_policy,
        optimizer_value=optimizer_value,
        states=states,
        actions=actions,
        returns=returns,
        advantages=advantages,
        l2_reg=L2_REG,
    )


def a2c_main():
    """A2C ALGORITHM"""
    # log training date
    localtime = time.asctime(time.localtime(time.time()))
    with open("assets/training_times/a2c_algorithm/training_time.txt", "a") as f:
        f.write(localtime)
        f.write("----------\n")

    # list of tau / lambda value: a parameter γ that allows us to reduce variance by downweighting rewards 
    # corresponding to delayed effects, at the cost of introducing bias
    tau_list = [0.50, 0.70, 0.90, 0.95, 0.97, 0.99]
    color_list = ["black", "red", "yellow", "green", "darkblue", "orange"]

    # plot
    plot = plt.figure()
    subplot = plot.add_subplot()

    for i in range(len(tau_list)):
        t0 = time.time()  # for logging training time

        # plot legend
        plot.legend(loc="upper right")

        # plot
        xval, yval = [], []
        plt.xlabel("Number Episodes")
        plt.ylabel("Rewards")
        plt.title(
            "Bipedal Walker v2 with A2C_GAE\ngamma=0.99, num_episodes=1000,\nL2_reg=1e-3, min_batch_size=2048"
        )
        string_label = "λ/tau = " + str(tau_list[i])
        (plotLine,) = subplot.plot(xval, yval, color_list[i], label=string_label)
        subplot.set_xlim([0, MAX_NUM_ITER])
        subplot.set_ylim([-300, 400])

        # run iteration/episode
        for i_iter in range(MAX_NUM_ITER):
            # gather and store (st,at,rt,s′t) by acting in the environment using the current policy
            batch, log = agent.collect_samples(MIN_BATCH_SIZE, RENDER)

            # update the parameter for each time step - second for loop
            update_a2c_params(batch, tau_list[i])

            if i_iter % LOG_INTERVAL == 0:
                print(
                    f'Episode {i_iter+1} finished. Highest reward: {log["max_reward"]}'
                )

            # if agent was able to get the highest rewards of 300, then log the episode number
            if log["max_reward"] >= 300.0:
                with open(
                    "assets/log_episodes_300_rewards/a2c_algorithm/log_300.txt", "a"
                ) as f:
                    f.write(
                        "Episode "
                        + str(i_iter + 1)
                        + " of lambda/tau "
                        + str(tau_list[i])
                        + " has the reward of "
                        + str(log["max_reward"])
                        + ".\n"
                    )
                    f.close()

            # append plot
            xval.append(i_iter)
            yval.append(log["max_reward"])
            plotLine.set_xdata(xval)
            plotLine.set_ydata(yval)
            plot.savefig("./results/a2c_max_reward")

            # optional: save trained model
            if SAVE_MODEL_INTERVAL > 0 and (i_iter + 1) % SAVE_MODEL_INTERVAL == 0:
                to_device(torch.device("cpu"), policy_net, value_net)

                pickle.dump(  # write the trained model to folder
                    (policy_net, value_net, running_state),
                    open(
                        os.path.join(
                            saved_assets_dir(),
                            "learned_models/a2c_algorithm/bipedal_walker_v2_a2c_cpu.p",
                        ),
                        "wb",
                    ),
                )
                to_device(device, policy_net, value_net)

            # clean up gpu memory after every iteration
            torch.cuda.empty_cache()

        t1 = time.time()
        print(f"All episodes finished. Training time of A2C is: {t1-t0}")

        # write training time to file
        with open("assets/training_times/a2c_algorithm/training_time.txt", "a") as f:
            f.write(
                "- The training time for 1000 episode of A2C_GAE with the lambda-return/tau-return of "
            )
            f.write(str(tau_list[i]))
            f.write(" is: ")
            f.write(str(t1 - t0))
            f.write(" seconds.\n")
            f.close()
