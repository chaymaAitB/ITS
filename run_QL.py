import cityflow
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import math

# =========================
# CONFIG
# =========================
CONFIG_PATH = "config.json"
STEPS_PER_EPISODE = 16000
LOG_EVERY = 100

EPISODES = 50
EPISODES_AVG = 5

episode_rewards = []
avg_rewards = []

current_phase = 0
node5_phase = 0
node5_counter = 0

# Track best episode

best_episode = -1
best_reward = -float("inf")

best_step_history = []
best_reward_history = []
best_queue_history = []

# =========================
# Q-LEARNING TABLE
# =========================
Q_table = {}

# =========================
# Variables
# =========================
NUM_PHASES = 4   # or  depending on your roadnet

# =========================
# RL PARAMETERS
# =========================
ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
# Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random
# start moderate exploration

EPSILON_MAX = 0.3
EPSILON_MIN = 0.05
EPSILON = EPSILON_MAX
EPSILON_DECAY = 0.995

# ================================
# Additional Stability Parameters 
# ================================

MIN_GREEN_STEPS = 100

# =========================
# NODE 5 FIXED-TIME CONTROL
# =========================

NODE5_PHASES = [0, 1, 2, 3]
NODE5_TIMES = [10, 3, 42, 3]



def fixed_time_node5(step, engine):
    global node5_phase, node5_counter

    node5_counter += 1

    if node5_counter >= NODE5_TIMES[node5_phase]:
        node5_counter = 0
        node5_phase = (node5_phase + 1) % 4

        engine.set_tl_phase("Node5", node5_phase)

# =========================
# HELPER FUNCTIONS
# =========================

def get_max_Q_value_of_state(state): #1. Objective Function 1

    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))

    return np.max(Q_table[state])

def get_reward(state): #2. Constraint 2
    return -(state[0] + state[1])

def get_state(): #3.& 4.& 8. Constraint 3 & 4 & 8
    global current_phase

    lane_counts = engine.get_lane_waiting_vehicle_count()

    lanes_EW = [
        "Node1_2_EB_0",
        "Node1_2_EB_1",
        "Node1_2_EB_2",
        "Node2_3_WB_0",
        "Node2_3_WB_1",
        "Node2_3_WB_2"
    ]

    queue_EW = sum(lane_counts[l] for l in lanes_EW)

    lanes_NS = [
        "Node2_7_SB_0",
        "Node2_7_SB_1",
        "Node2_7_SB_2",
        "Node2_5_NB_0",
        "Node2_5_NB_1",
        "Node2_5_NB_2"
    ]

    queue_NS = sum(lane_counts[l] for l in lanes_NS)

    return (queue_EW, queue_NS, current_phase)

def apply_action(action, step, tls_id="Node2"): #5. Constraint 5

    global current_phase
    global last_switch_step

    if action == 0:
        return current_phase

    if step - last_switch_step >= MIN_GREEN_STEPS:

        # Toggle between green phases only
        if current_phase == 0:
            current_phase = 2
        else:
            current_phase = 0

        engine.set_tl_phase(tls_id, current_phase)

        last_switch_step = step
    return current_phase

def update_Q_table(old_state, action, reward, new_state): #6. Constraint 6

    if old_state not in Q_table:
        Q_table[old_state] = np.zeros(len(ACTIONS))

    # 1) Predict current Q-values from old_state (current state)
    old_q = Q_table[old_state][action]

    # 2) Predict Q-values for new_state to get max future Q (new state)
    best_future_q = get_max_Q_value_of_state(new_state)

    # 3) Incorporate ALPHA to partially update the Q-value and update Q table
    Q_table[old_state][action] = (
        old_q
        + ALPHA * (reward + GAMMA * best_future_q - old_q)
    )

def get_action_from_policy(state): #7. Constraint 7

    if state not in Q_table:
        Q_table[state] = np.zeros(len(ACTIONS))

    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    return int(np.argmax(Q_table[state]))

# =========================
# TRAINING LOOP
# =========================
start_time = time.time()

print("\n=== Starting CityFlow Q-Learning with Episodes ===")

for episode in range(EPISODES):

    # reset environment per episode
    engine = cityflow.Engine(CONFIG_PATH, thread_num=8)
    
    # Reset episode variables
    current_phase = 0

    cumulative_reward = 0.0

    last_switch_step = -MIN_GREEN_STEPS

    node5_phase = 0
    node5_counter = 0


    print(f"\n--- Episode {episode + 1}/{EPISODES} ---")
    engine.set_tl_phase("Node2", current_phase)
    
    # =========================
    # STORAGE FOR PLOTS
    # =========================
    ep_step_history = []
    ep_reward_history = []
    ep_queue_history = []

    for step in range(STEPS_PER_EPISODE):

        # =========================
        # NODE 2 (RL CONTROLLED)
        # =========================
        # current_simulation_step = step
        
        # 1. observe current state
        state = get_state()

        # 2. choose action
        action = get_action_from_policy(state)

        # 3. apply RL action (Node2 only)
        apply_action(action, step)

        # =========================
        # NODE 5 (FIXED CONTROL)
        # =========================

        # 4. apply fixed control (Node5)
        fixed_time_node5(step, engine)

        # 5. advance simulation
        engine.next_step()
        
        # 6. observe new state
        new_state = get_state()

        reward = get_reward(new_state)
        cumulative_reward += reward

        # 7. update Q-table
        update_Q_table(state, action, reward, new_state)

        # Record data every 100 steps
        if step % 100 == 0:
            ep_step_history.append(step)
            ep_reward_history.append(cumulative_reward)
            ep_queue_history.append(new_state[0] + new_state[1])

    # Exploration rate decay
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    episode_rewards.append(cumulative_reward)

    print(f"Episode {episode} Reward: {cumulative_reward:.2f}")
    
    #  Track best episode
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        best_episode = episode

        # overwrite best logs
        best_step_history = ep_step_history
        best_reward_history = ep_reward_history
        best_queue_history = ep_queue_history

        print(f"🔥 New BEST episode: {episode} | reward = {best_reward:.2f}")
    # average per EPISODES_AVG episodes
    if (episode + 1) % EPISODES_AVG == 0:
        avg = np.mean(episode_rewards[-EPISODES_AVG:])
        avg_rewards.append(avg)
        print(f"\n>>> Avg reward (last {EPISODES_AVG} episodes): {avg:.2f}")
    

# =========================
# CLOSE SIMULATION
# =========================

end_time = time.time()

training_time = end_time - start_time

print("Simulation finished.")

print("\n\n=== TRAINING SUMMARY ===")
print(f"Episodes: {EPISODES}")
print(f"Steps per episode: {STEPS_PER_EPISODE}")
print(f"Total simulation steps: {EPISODES * STEPS_PER_EPISODE}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Training time: {training_time/60:.2f} minutes")

# Print final Q-table info
#print("\nOnline Training completed. Final Q-table size:", len(Q_table))
#for st, actions in Q_table.items():
#    print("State:", st, "-> Q-values:", actions)
# =========================
# Visualization of Results
# =========================

# Create one figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------
# Plot 1: Cumulative Reward
# -------------------------
axes[0].plot(best_step_history, best_reward_history, marker='o', linestyle='-')
axes[0].set_xlabel("Simulation Step")
axes[0].set_ylabel("Cumulative Reward")
axes[0].set_title("QL: Cumulative Reward")
axes[0].grid(True)

# -------------------------
# Plot 2: Queue Length
# -------------------------
axes[1].plot(best_step_history, best_queue_history, marker='o', linestyle='-')
axes[1].set_xlabel("Simulation Step")
axes[1].set_ylabel("Total Queue Length")
axes[1].set_title("QL: Queue Length")
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("plots/QL_results.png")

# ------------------------------
# Plot 3: Episode Reward Trend
# ------------------------------

plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Reward Trend")
plt.grid(True)
plt.savefig("plots/QL_episode_rewards.png")
