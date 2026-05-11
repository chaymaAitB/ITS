# ITS Study Case: Node 2 Traffic Control
![Description](figures/archroads.png)

This project implements and compares **DQN**, **QL (Q-Learning)**, and **Fixed Timing (FT)** for traffic signal control at **Node 2** in a synthetic traffic network.

### Network & Flow Configuration

The traffic network consists of **6 nodes** with the following signalized movements at **Node 2**:

| Movement | Direction |
|----------|-----------|
| Node 1→2 | Westbound (WB) |
| Node 1→2 | Eastbound (EB) |
| Node 2→5 | Southbound (SB) |
| Node 2→5 | Northbound (NB) |
| Node 2→3 | Westbound (WB) |
| Node 2→3 | Eastbound (EB) |

### Training Environment

- **Simulator**: CityFlow
- **Training Logic**: SUMO-compatible flow configuration
- **Objective**: Minimize queue length and maximize cumulative reward

> Training was conducted using CityFlow as the microscopic traffic simulator, with network and flow definitions adapted from SUMO test cases.

## Fixing Timing
 ![Description](figures/FT_results.png)

- Steps: 16,000
- Training time: 19.14 sec (0.32 min)
- Episode reward: **-32,953.57**

Fixed policy, no adaptation → low reward, high queue length.

---
## Q-learning
### Agent Workflow

- **State** → Detected vehicles + current traffic light phase  
- **Action** → Switch traffic light  
- **Reward** → Maximize cumulative reward  
- **Goal** → Learn optimal policy via AI algorithm (DQN / QL)
  
![Description](figures/agent.png)

![Description](figures/ql.png)

## QL Performance

![Description](figures/QL_results.png)

 === TRAINING SUMMARY QL ===

- Episodes: 50
- Steps per episode: 16,000
- Total interaction steps: 800,000
- Training time: 8.17 min

Cumulative reward reaches -15,000 to -20,000. Queue length fluctuates between 30–50.


## DQN
### DQN Neural Network Architecture

![Description](figures/dqn.png)

- **Input Layer** → North-South queue length, East-West queue length, Signal phase
- **Hidden Layers** → 2 hidden layers
- **Activation** → ReLU (Rectified Linear Unit)
- **Loss Function** → MSE (Mean Squared Error)
- **Optimizer** → Adam (learning_rate = 0.001)
- **Output Layer** → Action (switch traffic light)

> **Note**: Simple 3-input architecture processes traffic state and outputs optimal signal switching decisions.

![Description](figures/dqn_arch.jpeg)

###  DQN Performance
![Description](figures/DQN_results.png)

- Episodes: 3
- Steps per episode: 12,000
- Total steps: 36,000
- Training time: 2,562.48 sec (42.71 min)

Learns effective policy with cumulative reward stabilizing around -5,000 to -10,000 after convergence. Queue length kept mostly below 20.

