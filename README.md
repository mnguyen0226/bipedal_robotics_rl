# Performance Comparison and Analysis Between Q-Learning, A2C with Generalized Advantage Estimation, and PPO with Generalized Advantage Estimation in BipedalWalker-v2 - ECE 5984

In recent years, reinforcement learning (RL) algorithms have been implemented in several robotics and control systems applications. Several RL techniques are used to achieve basic autonomous controls, path-findings, vision tracker, and intelligent decision. Stabilizing bipedal walking robot is one of the challenging problems. In this paper, I will experiment and evaluate the three reinforcement learning algorithms to solve the simulated bipedal walking problem. Without any prior knowledge of its surrounding environment, the agent is able to demonstrate successful walking ability through trial and error via Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO). The results show that A2C and PPO with different bias estimation rates are capable of solving the bipedal walking problem.

## Reproducibility

## Q Learning Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/q_learning_max_reward_gpu.png)

## A2C with GAE Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/a2c_max_reward_gpu.png)

## PPO with GAE Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/ppo_max_reward_gpu.png)

## Proposal & Report
- [Proposal](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/docs/Project%20Proposal.pdf)
- Final Paper

## References

(aia) 
nguye@DESKTOP-OBHI23I MINGW64 ~/OneDrive/Desktop/Senior/ECE 5984 Reinforcement Learning/rl_value_based_vs_value_policy_based (main)
$ python src/main.py 


