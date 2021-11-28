# Implementation of Bipedal Walking with Q Learning, A2C, A2C GAE, PPO
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/20/2021

from algorithms.a2c_algorithm.a2c_main import a2c_main
from algorithms.ppo_algorithm.ppo_main import ppo_main
from algorithms.q_learning_algorithm.q_learning_main import q_learning_main


def main():
    # q_learning_main()
    # a2c_main()
    ppo_main()


if __name__ == "__main__":
    main()
