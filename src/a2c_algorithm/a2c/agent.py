import torch
import math
import time
import multiprocessing
from utils import to_device
from utils import collect_samples
from utils import merge_log
import numpy as np


class Agent:
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
        self.env = env
        self.policy = policy
        self.device = device
        self.custom_reward = custom_reward
        self.mean_action = mean_action
        self.running_state = running_state
        self.render = render
        self.num_threads = num_threads

    def collect_samples(self, min_batch_size, render):
        self.render = render
        t_start = time.time()
        to_device(torch.device("cpu"), self.policy)
        thread_batch_size = int(math.floor(min_batch_size / self.num_threads))
        queue = multiprocessing.Queue()
        workers = []

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

        memory, log = collect_samples(
            0,
            None,
            self.env,
            self.policy,
            self.custom_reward,
            self.mean_action,
            self.render,
            self.running_state,
            thread_batch_size,
        )

        worker_logs = [None] * len(workers)
        worker_memories = [None] * len(workers)
        for _ in workers:
            pid, worker_memory, worker_log = queue.get()
            worker_memories[pid - 1] = worker_memory
            worker_logs[pid - 1] = worker_log
        for worker_memory in worker_memories:
            memory.append(worker_memory)
        batch = memory.sample()
        if self.num_threads > 1:
            log_list = [log] + worker_logs
            log = merge_log(log_list)
        to_device(self.device, self.policy)
        t_end = time.time()
        log["sample_time"] = t_end - t_start
        log["action_mean"] = np.mean(np.vstack(batch.action), axis=0)
        log["action_min"] = np.min(np.vstack(batch.action), axis=0)
        log["action_max"] = np.max(np.vstack(batch.action), axis=0)
        return batch, log
