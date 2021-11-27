# Implementation of Bipedal Walking with Q Learning
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/20/2021

import numpy as np
import gym
import matplotlib.pyplot as plt
from collections import defaultdict
from algorithms.q_learning_algorithm import q_learning


def q_learning_main():
    """User Interface"""
    # initialize hyperparameters
    highest_reward = -300
    num_episodes = 1000  # number of episode
    gamma = 0.99  # discount factor
    alpha = 0.001  # learning rate

    env = gym.make("BipedalWalker-v2")
    q_table = defaultdict(lambda: np.zeros((10, 10, 10, 10)))

    # plot
    plot = plt.figure()
    xval, yval = [], []
    subplot = plot.add_subplot()
    plt.xlabel("Number Episodes")
    plt.ylabel("Rewards")
    plt.title("Rewards vs Number Episodes")
    (plotLine,) = subplot.plot(xval, yval)
    subplot.set_xlim([0, num_episodes])
    subplot.set_ylim([-220, -80])

    for i in range(1, num_episodes + 1):
        # collect rewards and highest reward
        curr_episode_reward, highest_reward = q_learning(
            env=env,
            num_episode=i,
            q_table=q_table,
            highest_reward=highest_reward,
            alpha=alpha,
            gamma=gamma,
        )
        print(f"Episode {i} finished. Highest reward: {highest_reward}")

        # append plot
        xval.append(i)
        yval.append(curr_episode_reward)
        plotLine.set_xdata(xval)
        plotLine.set_ydata(yval)
        plot.savefig("./results/q_learning_max_reward")

    print(
        "All episodes finished. Highest reward per episode achieved: "
        + str(highest_reward)
    )


if __name__ == "__main__":
    q_learning_main()
