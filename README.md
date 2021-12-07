# Performance Comparison and Analysis Between Q-Learning, A2C with Generalized Advantage Estimation, and PPO with Generalized Advantage Estimation in BipedalWalker-v2 - ECE 5984

## Proposal & Report

## Bipedal Walker - v2 Environment

## Q Learning

## A2C - Advantage Actor Critic

## A2C GAE - Advantage Actor Critic Generalized Advantage Estimation

## PPO - Proximal Policy Optimization

## References

(aia) 
nguye@DESKTOP-OBHI23I MINGW64 ~/OneDrive/Desktop/Senior/ECE 5984 Reinforcement Learning/rl_value_based_vs_value_policy_based (main)
$ python src/main.py 

Plan for results:
"The asynchronous advantage actor-critic method could be potentially improved by using other ways of estimating the advantage function, 
such as generalized advantage estimation of (Schulmanet al., 2015b)."

5000 timesteps

- best reward / episode: Q Learning: [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9] Learning Rate 
- best reward / episode: A2C + GAE: different Lamda - Tau  [0.50, 0.70, 0.90, 0.95, 0.97, 0.99] - Lamda Grid Search
- best reward / episode: PPO + GAE:  [0.50, 0.70, 0.90, 0.95, 0.97, 0.99] Lamda Grid Search
# if statement that log for reward > 300

=> Save graphs + Render video on local machine (pick the gae that has the highest trend)

- Discuss the advantage and disadvantage of the three equation
- Compare with top benchmark AI - Plot number of episode / ? (https://github.com/openai/gym/wiki/Leaderboard)
- Compare result between each other and between DDPG algorithms project: http://arxiv-export-lb.library.cornell.edu/pdf/1807.05924 _ IEEE

COMPARISON METRICS: Rewards/Episodes (Graph), Timer (log), Number of episode that reach 300 rewards 
- CHANGE 300 log , 5000 time step

- Paper: 
https://www.overleaf.com/latex/templates/neurips-2021/bfjnthbqvhgs
https://www.overleaf.com/latex/templates/icml2021-template/dsftnbmjgyhv

IDEA: May Implement TRPO if have time

## Recording:
https://drive.google.com/drive/folders/1adlMlMAl7jwFxOLdTJnXYQfEmJs8Pv8E?usp=sharing
