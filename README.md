# Performance Comparison and Analysis Between Q-Learning, A2C with Generalized Advantage Estimation, and PPO with Generalized Advantage Estimation in BipedalWalker-v2

## About
In recent years, reinforcement learning (RL) algorithms have been implemented in several robotics and control systems applications. Several RL techniques are used to achieve basic autonomous controls, path-findings, vision tracker, and intelligent decision. Stabilizing bipedal walking robot is one of the challenging problems. In this paper, I will experiment and evaluate the three reinforcement learning algorithms to solve the simulated bipedal walking problem. Without any prior knowledge of its surrounding environment, the agent is able to demonstrate successful walking ability through trial and error via Advantage Actor-Critic (A2C) and Proximal Policy Optimization (PPO). The results show that A2C and PPO with different bias estimation rates are capable of solving the bipedal walking problem.

![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/docs/image/bipedal_wallpaper.png)

## Reproducibility
- Fork the project and enter the directory: `$cd soo_non_convex_ml`
- `$cd rl_value_based_vs_value_policy_based`
- To run experiments on all three algorithms Q-Learning, A2C, PPO: `python src/main.py`
- To specify algorithms in `src/main.py`:
```python
# train q leanring
q_learning_main()

# train a2c gae
# a2c_main()

# train ppo gae
# ppo_main()
```

## Recorded Agents
- [Link](https://drive.google.com/drive/folders/1adlMlMAl7jwFxOLdTJnXYQfEmJs8Pv8E?usp=sharing)

## Q-Learning Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/q_learning_max_reward_gpu.png)

## A2C with GAE Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/a2c_max_reward_gpu.png)

## PPO with GAE Performance
![alt text](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/results/gpu_trained/ppo_max_reward_gpu.png)

## Proposal & Report
- [Proposal](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/docs/Project%20Proposal.pdf)
- [Final Paper](https://github.com/mnguyen0226/rl_value_based_vs_value_policy_based/blob/main/docs/Reinforcement%20Learning%20Final%20Paper.pdf)

## References
- [1] Pieter Abbeel. L3 Policy Gradients and Advantage Estimation (Foundations of Deep RL Series). 2021. URL: https://www.youtube.com/watchv=AKbX1Zvo7r8&ab_channel=PieterAbbeel.
- [2] Pieter Abbeel. L4 TRPO and PPO (Foundations of Deep RL Series). 2021. URL: https://www.youtube.com/watch?v=KjWF8VIMGiY&ab_channel=PieterAbbeel.
- [3] Evan Ackerman. Bipedal Robots Are Learning To Move With Arms as Well as Legs. 2021. URL: https://spectrum.ieee.org/bipedal-robot-learning-to-move-arms-legs.
- [4] Boston Dynamics’s Atlas. URL: https://www.bostondynamics.com/atlas.
- [5] G. Brockman et al. Openai Gym. 2016. URL: https://arxiv.org/abs/1606.01540.
- [6] Chris and Mandy. Exploration Vs. Exploitation - Learning The Optimal Reinforcement Learning Policy. 2018. URL: https://deeplizard.com/learn/video/mo96Nqlo1L8.
- [7] DanielGörges. “Relations between Model Predictive Control and Reinforcement Learning”.In: IFAC-PapersOnLine 50.1 (2017), pp. 4920–4928.
- [8] Laura Graesser and Wah Loon Keng. Foundations of Deep Reinforcement Learning: Theory and Practice in Python. 2018. URL: https : / / slm - lab . gitbook . io / slm - lab /publications-and-talks/instruction-for-the-book-+-intro-to-rl-section.
- [9] Alexander Van de Kleut. Actor-Critic Methods, Advantage Actor-Critic (A2C) and Generalized Advantage Estimation (GAE). 2020. URL: https://avandekleut.github.io/a2c.
- [10] Alexander Van de Kleut. Beyond vanilla policy gradients: Natural policy gradients, trust region policy optimization (TRPO) and Proximal Policy Optimization (PPO). 2021. URL: https://avandekleut.github.io/ppo.
- [11] Jens Kober, J. Andrew Bagnell, and Jan Peters. Reinforcement Learning in Robotics: A Survey. 2013. URL: https://www.ri.cmu.edu/pub_files/2013/7/Kober_IJRR_2013.pdf.
- [12] Russ Mitchell. Two die in driverless Tesla incident. Where are the regulators? 2021. URL: https://www.latimes.com/business/story/2021-04-19/tesla-on-autopilotkills-two-where-are-the-regulators.
- [13] V. Mnih et al. Asynchronous methods for deep reinforcement learning. 2016. URL: https://arxiv.org/abs/1602.01783v2.
- [14] V. Mnih et al. Playing Atari with deep reinforcement learning. 2016. URL: https://arxiv.org/abs/1602.01783v2.
- [15] OpenAI’s Gym BipedalWalker-v2. URL: https://gym.openai.com/envs/BipedalWalker-v2/.
- [16] J. Schulman et al. High-dimensional continuous control using generalized advantage estimation. 2018. URL: https://arxiv.org/abs/1506.02438.
- [17] J. Schulman et al. Proximal policy optimization algorithms. 2017. URL: https://arxiv.
org/abs/1707.06347.
- [18] R.S Sutton and A.G Barto. Reinforcement Learning: An Introduction. 2018.



