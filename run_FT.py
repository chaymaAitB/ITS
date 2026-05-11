import cityflow
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# =========================
# CONFIG
# =========================
CONFIG_PATH = "config.json"
TOTAL_STEPS = 16000
LOG_EVERY = 100

# =========================
# ENGINE INIT
# =========================
engine = cityflow.Engine(CONFIG_PATH, thread_num=8)

# =========================
# Q-LEARNING TABLE
# =========================
Q_table = {}

# =========================
# STORAGE FOR PLOTS
# =========================
step_history = []
reward_history = []
queue_history = []

cumulative_reward = 0.0

# =========================
# Variables
# =========================
current_phase = 0
NUM_PHASES = 4   # or  depending on your roadnet

# =========================
# RL PARAMETERS
# =========================
ACTIONS = [0, 1]       # The discrete action space (0 = keep phase, 1 = switch phase)

ALPHA = 0.1            # Learning rate (α) between[0, 1]    #If α = 1, you fully replace the old Q-value with the newly computed estimate.
                                                            #If α = 0, you ignore the new estimate and never update the Q-value.
GAMMA = 0.9            # Discount factor (γ) between[0, 1]  #If γ = 0, the agent only cares about the reward at the current step (no future rewards).
                                                            #If γ = 1, the agent cares equally about current and future rewards, looking at long-term gains.
EPSILON = 0.0          # Exploration rate (ε) between[0, 1] #If ε = 0 means very greedy, if=1 means very random

# ================================
# Additional Stability Parameters 
# ================================

MIN_GREEN_STEPS = 0
last_switch_step = -MIN_GREEN_STEPS

# =========================
# HELPER FUNCTIONS
# =========================

def get_state():
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


def get_reward(state): #2. Constraint 2
    return -math.log(1 + state[0] + state[1])


# =========================
# TRAINING LOOP
# =========================
start_time = time.time()

print("Starting CityFlow Fixing Timing...")
# print(dir(engine))
for step in range(TOTAL_STEPS):
    current_simulation_step = step

    state = get_state()

    engine.next_step()

    new_state = get_state()

    reward = get_reward(new_state)

    cumulative_reward += reward

    # Record data every 100 steps
    if step % 100 == 0:
        print(f"Step {step}, Current_State: {state}, New_State: {new_state}, Reward: {reward:.2f}, Cumulative Reward: {cumulative_reward:.2f}")
        step_history.append(step)
        reward_history.append(cumulative_reward)
        queue_history.append(sum(new_state[:-1]))  # sum of queue lengths
        print("Current Q-table:")
        for st, qvals in Q_table.items():
            print(f"  {st} -> {qvals}")


# =========================
# CLOSE SIMULATION
# =========================
print("Simulation finished.")

end_time = time.time()

training_time = end_time - start_time

print("Simulation finished.")

print("\n\n=== TRAINING SUMMARY ===")
print(f"Steps: {TOTAL_STEPS}")
print(f"Training time: {training_time:.2f} seconds")
print(f"Training time: {training_time/60:.2f} minutes")

# =========================
# Visualization of Results
# =========================

# Create one figure with 2 subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# -------------------------
# Plot 1: Cumulative Reward
# -------------------------
axes[0].plot(step_history, reward_history, marker='o', linestyle='-')
axes[0].set_xlabel("Simulation Step")
axes[0].set_ylabel("Cumulative Reward")
axes[0].set_title("Fixed Timing: Cumulative Reward")
axes[0].grid(True)

# -------------------------
# Plot 2: Queue Length
# -------------------------
axes[1].plot(step_history, queue_history, marker='o', linestyle='-')
axes[1].set_xlabel("Simulation Step")
axes[1].set_ylabel("Total Queue Length")
axes[1].set_title("Fixed Timing: Queue Length")
axes[1].grid(True)

# Adjust layout
plt.tight_layout()

# Save figure
plt.savefig("plots/FT_results.png")

# Show figure
plt.show()