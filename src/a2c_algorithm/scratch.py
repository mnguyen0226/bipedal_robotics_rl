# Implementation of Bipedal Walking with A2C - Advantage Actor Critics
# By Minh Nguyen
# ECE 5984 - Reinforcement Learning
# 11/21/2021

import argparse
import gym
import torch
import os
import sys
import pickle
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.additional_torch import to_device
from models import Policy
from models import Value
from models import DiscretePolicy
from a2c import a2c_step
from a2c import estimate_advantages
from a2c import Agent
import numpy as np
from utils.zfilter import ZFilter

parser = argparse.ArgumentParser(description="PyTorch A2C example")
parser.add_argument(
    "--env-name",
    default="BipedalWalker-v2",
    metavar="G",  # Hopper-v2 CartPole-v1 BipedalWalker-v2
    help="name of the environment to run",
)
parser.add_argument(
    "--model-path",
    metavar="G",
    default=None,  # os.path.join(assets_dir(), "learned_models/BipedalWalker-v2_a2c.p"),
    help="path of pre-trained model",
)
parser.add_argument(
    "--render", action="store_true", default=False, help="render the environment"
)
parser.add_argument(
    "--log-std",
    type=float,
    default=-1.0,
    metavar="G",
    help="log std for the policy (default: -1.0)",
)
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99,
    metavar="G",
    help="discount factor (default: 0.99)",
)
parser.add_argument(
    "--tau", type=float, default=0.95, metavar="G", help="gae (default: 0.95)"
)
parser.add_argument(
    "--l2-reg",
    type=float,
    default=1e-3,
    metavar="G",
    help="l2 regularization regression (default: 1e-3)",
)
parser.add_argument(
    "--num-threads",
    type=int,
    default=1,
    metavar="N",
    help="number of threads for agent (default: 4)",
)
parser.add_argument(
    "--seed", type=int, default=1, metavar="N", help="random seed (default: 1)"
)
parser.add_argument(
    "--min-batch-size",
    type=int,
    default=2048,
    metavar="N",
    help="minimal batch size per A2C update (default: 2048)",
)
parser.add_argument(
    "--max-iter-num",
    type=int,
    default=50000,
    metavar="N",
    help="maximal number of main iterations (default: 500)",
)
parser.add_argument(
    "--log-interval",
    type=int,
    default=1,
    metavar="N",
    help="interval between training status logs (default: 1)",
)
parser.add_argument(
    "--save-model-interval",
    type=int,
    default=100,
    metavar="N",
    help="interval between saving model (default: 0, means don't save)",
)
parser.add_argument("--gpu-index", type=int, default=0, metavar="N")
args = parser.parse_args()
################
dtype = torch.float64
torch.set_default_dtype(dtype)
device = (
    torch.device("cuda", index=args.gpu_index)
    if torch.cuda.is_available()
    else torch.device("cpu")
)
if torch.cuda.is_available():
    torch.cuda.set_device(args.gpu_index)

"""environment"""
env = gym.make(args.env_name)
state_dim = env.observation_space.shape[0]
is_disc_action = len(env.action_space.shape) == 0
running_state = ZFilter((state_dim,), clip=5)
# running_reward = ZFilter((1,), demean=False, clip=10)

"""seeding"""
np.random.seed(args.seed)
torch.manual_seed(args.seed)
env.seed(args.seed)

"""define actor and critic"""
if args.model_path is None:
    if is_disc_action:
        print("DISCRETE POLICY\n")
        policy_net = DiscretePolicy(state_dim, env.action_space.n)
    else:
        print("POLICY\n")
        policy_net = Policy(state_dim, env.action_space.shape[0], log_std=args.log_std)
    value_net = Value(state_dim)
else:
    policy_net, value_net, running_state = pickle.load(open(args.model_path, "rb"))
policy_net.to(device)
value_net.to(device)

optimizer_policy = torch.optim.Adam(policy_net.parameters(), lr=4e-4)
optimizer_value = torch.optim.Adam(value_net.parameters(), lr=8e-4)

"""create agent"""
agent = Agent(
    env,
    policy_net,
    device,
    running_state=running_state,
    render=args.render,
    num_threads=args.num_threads,
)


def update_params(batch):
    states = torch.from_numpy(np.stack(batch.state)).to(dtype).to(device)
    actions = torch.from_numpy(np.stack(batch.action)).to(dtype).to(device)
    rewards = torch.from_numpy(np.stack(batch.reward)).to(dtype).to(device)
    masks = torch.from_numpy(np.stack(batch.mask)).to(dtype).to(device)
    with torch.no_grad():
        values = value_net(states)

    """get advantage estimation from the trajectories"""
    advantages, returns = estimate_advantages(
        rewards, masks, values, args.gamma, args.tau, device
    )

    """perform TRPO update"""
    a2c_step(
        policy_net,
        value_net,
        optimizer_policy,
        optimizer_value,
        states,
        actions,
        returns,
        advantages,
        args.l2_reg,
    )


def main_loop():
    for i_iter in range(args.max_iter_num):
        render = False
        """generate multiple trajectories that reach the minimum batch_size"""
        batch, log = agent.collect_samples(args.min_batch_size, render)
        t0 = time.time()
        update_params(batch)
        t1 = time.time()

        if i_iter % args.log_interval == 0:
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

        if (
            args.save_model_interval > 0
            and (i_iter + 1) % args.save_model_interval == 0
        ):
            to_device(torch.device("cpu"), policy_net, value_net)

            # torch.save(policy_net.state_dict(), 'actor_weights')
            # torch.save(value_net.state_dict(), 'critic_weights')

            pickle.dump(
                (policy_net, value_net, running_state),
                open(
                    os.path.join(
                        assets_dir(), "learned_models/{}_a2c.p".format(args.env_name)
                    ),
                    "wb",
                ),
            )
            to_device(device, policy_net, value_net)

        """clean up gpu memory"""
        torch.cuda.empty_cache()


main_loop()