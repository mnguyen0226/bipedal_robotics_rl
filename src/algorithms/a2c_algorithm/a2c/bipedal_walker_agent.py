import torch
import math
import time
import multiprocessing
from algorithms.utils import to_device
import numpy as np
from algorithms.utils import Memory

# Reference: https://github.com/pythonlessons/Reinforcement_Learning/blob/master/BipedalWalker-v3_PPO/BipedalWalker-v3_PPO.py
# Reference: https://slm-lab.gitbook.io/slm-lab/development/algorithms/a2c
# Reference: https://github.com/kengz/SLM-Lab


class BipedalWalkerAgent:
    def __init__(
        self,
        env,
        policy,
        device,
        custom_reward=None,
        mean_action=False,
        render=False,
        running_state=None,
        num_threads=1,
    ):
        """Constructor of BipedalWalkerAgent class

        Args:
            env: Bipedal Walker environment
            policy: Policy network
            device: CPU or GPU
            custom_reward: customed rewards collected from Critic network. Defaults to None.
            mean_action: mean values of action array. Defaults to False.
            render: allowance to render. Defaults to False.
            running_state: running state. Defaults to None.
            num_threads: number of thread run concurrently. Defaults to 1.
        """
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, render):
        """Collects batches of actions using parallel training if using GPU
        This is an override of collect_samples()

        Args:
            min_batch_size: size of mini batch
            render: render state set to False

        Returns:
            Batch of action and log
        """
        self.render = render  # set render
        t_start = time.time()  # record time
        to_device(torch.device("cpu"), self.policy)  # set cpu
        thread_batch_size = int(
            math.floor(min_batch_size / self.num_threads)
        )  # number of threads for batch size
        queue = (
            multiprocessing.Queue()
        )  # set parallel multi processing if able to run multicore
        workers = []  # number of worker

        for i in range(self.num_threads - 1):
            worker_args = (
                i + 1,
                queue,
                self.env,
                self.policy,
                self.custom_reward,
                self.mean_action,
                False,
                self.running_state,
                thread_batch_size,
            )
            workers.append(
                multiprocessing.Process(target=collect_samples, args=worker_args)
            )
        for worker in workers:
            worker.start()

        # recursive call collect_sample()
        memory, log = collect_samples(
            rand_init=0,
            queue=None,
            env=self.env,
            policy=self.policy,
            custom_reward=self.custom_reward,
            mean_action=self.mean_action,
            render=self.render,
            running_state=self.running_state,
            min_batch_size=thread_batch_size,
        )

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)

        for _ in workers:
            rand_init, worker_memory, worker_log = queue.get()
            worker_memories[rand_init - 1] = worker_memory
            worker_logs[rand_init - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()

        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = concat_log(log_list)

        to_device(self.device, self.policy)
        t_end = time.time()

        # save sampling time, mean, min, max action values
        log["sample_time"] = t_end - t_start
        log["action_mean"] = np.mean(np.vstack(batch.action), axis=0)
        log["action_min"] = np.min(np.vstack(batch.action), axis=0)
        log["action_max"] = np.max(np.vstack(batch.action), axis=0)

        return batch, log


def collect_samples(
    rand_init,
    queue,
    env,
    policy,
    custom_reward,
    mean_action,
    render,
    running_state,
    min_batch_size,
):
    """Helper function - Collect batch of action and log depends on the batch size

    Args:
        rand_init: random initialization for reproducibility
        queue: queue
        env: Bipedal Walker v2
        policy: Policy network
        custom_reward: customed reward
        mean_action: mean of action values
        render: allowance for render
        running_state: running state
        min_batch_size: mini batch size

    Returns:
        Memory array and log array
    """
    torch.randn(rand_init)
    log = dict()
    memory = Memory()
    num_steps = 0
    total_reward = 0
    min_reward = 1e6
    max_reward = -1e6
    total_c_reward = 0
    min_c_reward = 1e6
    max_c_reward = -1e6
    num_episodes = 0

    while num_steps < min_batch_size:  # execute while not exceed the min batch size
        state = env.reset()

        if running_state is not None:  # if there is running state
            state = running_state(state)

        reward_episode = 0  # initialize reward

        for t in range(10000):  # for 10000 time step
            state_var = torch.Tensor(state).unsqueeze(0)
            with torch.no_grad():
                if (
                    mean_action
                ):  # if there is a mean action value, then take action according to the policy network
                    action = policy(state_var)[0][0].numpy()
                else:
                    action = policy.select_action(state_var)[0].numpy()

            action = (
                int(action) if policy.is_disc_action else action.astype(np.float64)
            )  # convert action to int or float

            next_state, reward, done, _ = env.step(
                action
            )  # takle action amd cp;;ect omfp

            reward_episode += reward  # collect reward

            if running_state is not None:
                next_state = running_state(next_state)

            if custom_reward is not None:
                reward = custom_reward(state, action)
                total_c_reward += reward
                min_c_reward = min(min_c_reward, reward)
                max_c_reward = max(max_c_reward, reward)

            mask = 0 if done else 1

            memory.push(state, action, mask, next_state, reward)

            if render:
                env.render()
            if done:
                break

            state = next_state

        # update statisitics
        num_steps += t + 1
        num_episodes += 1
        total_reward += reward_episode
        min_reward = min(min_reward, reward_episode)
        max_reward = max(max_reward, reward_episode)

    # log statistic
    log["num_steps"] = num_steps
    log["num_episodes"] = num_episodes
    log["total_reward"] = total_reward
    log["avg_reward"] = total_reward / num_episodes
    log["max_reward"] = max_reward
    log["min_reward"] = min_reward

    if (
        custom_reward is not None
    ):  # if there is custom reward produced by the critic network
        log["total_c_reward"] = total_c_reward
        log["avg_c_reward"] = total_c_reward / num_steps
        log["max_c_reward"] = max_c_reward
        log["min_c_reward"] = min_c_reward

    if queue is not None:
        queue.put([rand_init, memory, log])
    else:
        return memory, log


def concat_log(log_list):
    """Helper funcition - merges saved log

    Args:
        log_list: list of log values

    Returns:
        Log array
    """
    log = dict()
    log["total_reward"] = sum([x["total_reward"] for x in log_list])
    log["num_episodes"] = sum([x["num_episodes"] for x in log_list])
    log["num_steps"] = sum([x["num_steps"] for x in log_list])
    log["avg_reward"] = log["total_reward"] / log["num_episodes"]
    log["max_reward"] = max([x["max_reward"] for x in log_list])
    log["min_reward"] = min([x["min_reward"] for x in log_list])
    if "total_c_reward" in log_list[0]:
        log["total_c_reward"] = sum([x["total_c_reward"] for x in log_list])
        log["avg_c_reward"] = log["total_c_reward"] / log["num_steps"]
        log["max_c_reward"] = max([x["max_c_reward"] for x in log_list])
        log["min_c_reward"] = min([x["min_c_reward"] for x in log_list])

    return log
