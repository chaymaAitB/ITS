# ITS

## Study Case
![Description](figures/archroads.png)

NOde 2 ,  creating network and flow in SOMU

TRaining in Cityflow 

## Fixing Timing
 ![Description](figures/FT_results.png)

- Steps: 16,000
- Training time: 19.14 sec (0.32 min)
- Episode reward: **-32,953.57**

Fixed policy, no adaptation → low reward, high queue length.

---
## Q-learning
![Description](figures/agent.png)

![Description](figures/ql.png)
 ![Description](figures/QL_results.png)

 === TRAINING SUMMARY QL ===

- Episodes: 50
- Steps per episode: 16,000
- Total interaction steps: 800,000
- Training time: 8.17 min

Cumulative reward reaches -15,000 to -20,000. Queue length fluctuates between 30–50.


## DQN

![Description](figures/dqn.png)

#### Model Rchitecture
![Description](figures/dqn_arch.jpeg)

 ![Description](figures/DQN_results.png)

 - Episodes: 3
- Steps per episode: 12,000
- Total steps: 36,000
- Training time: 2,562.48 sec (42.71 min)

Learns effective policy with cumulative reward stabilizing around -5,000 to -10,000 after convergence. Queue length kept mostly below 20.

