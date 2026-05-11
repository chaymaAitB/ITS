import cityflow
import random
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import math
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# =========================
# CONFIG
# =========================
CONFIG_PATH = "config.json"

STEPS_PER_EPISODE = 12000
EPISODES = 3

LOG_EVERY = 100

# =========================
# RL PARAMETERS
# =========================
ACTIONS = [0, 1]

ALPHA = 0.001
GAMMA = 0.9

EPSILON_MAX = 0.1
EPSILON_MIN = 0.05
EPSILON = EPSILON_MAX
EPSILON_DECAY = 0.995

MIN_GREEN_STEPS = 10

# =========================
# DQN PARAMETERS
# =========================
STATE_SIZE = 3
ACTION_SIZE = len(ACTIONS)

BATCH_SIZE = 64
MEMORY_SIZE = 20000
TARGET_UPDATE = 1000

memory = deque(maxlen=MEMORY_SIZE)

# =========================
# STORAGE
# =========================
episode_rewards = []

best_step_history = []
best_reward_history = []
best_queue_history = []

best_reward = -float("inf")

# =========================
# TIME START
# =========================
start_time = time.time()

# =========================
# MODEL
# =========================
def build_model():
    model = keras.Sequential([
        layers.Input(shape=(STATE_SIZE,)),
        layers.Dense(24, activation='relu'),
        layers.Dense(24, activation='relu'),
        layers.Dense(ACTION_SIZE, activation='linear')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=ALPHA),
        loss='mse'
    )
    return model

dqn_model = build_model()
target_model = build_model()
target_model.set_weights(dqn_model.get_weights())

# =========================
# STATE
# =========================
current_phase = 0
global_step = 0
last_switch_step = -MIN_GREEN_STEPS

def to_array(state):
    return np.array(state, dtype=np.float32).reshape(1, -1)

def get_reward(state):
    return -math.log(1 + state[0] + state[1])

def get_state(engine):
    global current_phase

    lane_counts = engine.get_lane_waiting_vehicle_count()

    lanes_EW = [
        "Node1_2_EB_0","Node1_2_EB_1","Node1_2_EB_2",
        "Node2_3_WB_0","Node2_3_WB_1","Node2_3_WB_2"
    ]

    lanes_NS = [
        "Node2_7_SB_0","Node2_7_SB_1","Node2_7_SB_2",
        "Node2_5_NB_0","Node2_5_NB_1","Node2_5_NB_2"
    ]

    queue_EW = sum(lane_counts[l] for l in lanes_EW)
    queue_NS = sum(lane_counts[l] for l in lanes_NS)

    return (queue_EW, queue_NS, current_phase)

# =========================
# ACTION POLICY (FAST)
# =========================
def get_action(state):
    global EPSILON

    if random.random() < EPSILON:
        return random.choice(ACTIONS)

    q_values = dqn_model(to_array(state), training=False).numpy()[0]
    return int(np.argmax(q_values))

# =========================
# MEMORY
# =========================
def store(s, a, r, s2, d):
    memory.append((s, a, r, s2, d))

# =========================
# TRAIN (OPTIMIZED)
# =========================
def train():
    if len(memory) < BATCH_SIZE:
        return

    batch = random.sample(memory, BATCH_SIZE)

    states = []
    targets = []

    for s, a, r, s2, d in batch:

        s_arr = to_array(s)
        s2_arr = to_array(s2)

        q = dqn_model(s_arr, training=False).numpy()[0]

        if d:
            q[a] = r
        else:
            future = np.max(target_model(s2_arr, training=False).numpy()[0])
            q[a] = r + GAMMA * future

        states.append(s_arr[0])
        targets.append(q)

    dqn_model.fit(
        np.array(states),
        np.array(targets),
        epochs=1,
        verbose=0
    )

# =========================
# ACTION APPLY
# =========================
def apply_action(engine, action, step):
    global current_phase, last_switch_step

    if action == 1 and step - last_switch_step >= MIN_GREEN_STEPS:
        current_phase = 2 if current_phase == 0 else 0
        engine.set_tl_phase("Node2", current_phase)
        last_switch_step = step

# =========================
# TRAINING LOOP
# =========================
print("\n=== STARTING TRAINING ===")

for episode in range(EPISODES):

    engine = cityflow.Engine(CONFIG_PATH, thread_num=8)

    current_phase = 0
    last_switch_step = -MIN_GREEN_STEPS

    engine.set_tl_phase("Node2", current_phase)

    cumulative_reward = 0

    ep_steps = []
    ep_rewards = []
    ep_queues = []

    print(f"\n--- Episode {episode+1}/{EPISODES} ---")

    for step in range(STEPS_PER_EPISODE):

        global_step += 1

        state = get_state(engine)
        action = get_action(state)

        apply_action(engine, action, step)

        engine.next_step()

        new_state = get_state(engine)
        reward = get_reward(new_state)

        cumulative_reward += reward

        done = (step == STEPS_PER_EPISODE - 1)

        store(state, action, reward, new_state, done)

        # 🔥 TRAIN EVERY 4 STEPS (IMPORTANT SPEED FIX)
        if global_step % 4 == 0:
            train()

        # target update
        if global_step % TARGET_UPDATE == 0:
            target_model.set_weights(dqn_model.get_weights())

        # logging
        if step % LOG_EVERY == 0:
            ep_steps.append(step)
            ep_rewards.append(cumulative_reward)
            ep_queues.append(new_state[0] + new_state[1])

            print(
                f"Ep {episode} | Step {step} | "
                f"State {state} | Action {action} | "
                f"Reward {reward:.2f}"
            )

    episode_rewards.append(cumulative_reward)

    # epsilon decay
    EPSILON = max(EPSILON_MIN, EPSILON * EPSILON_DECAY)

    # best episode tracking
    if cumulative_reward > best_reward:
        best_reward = cumulative_reward
        best_step_history = ep_steps
        best_reward_history = ep_rewards
        best_queue_history = ep_queues

    print(f"Episode reward: {cumulative_reward:.2f}")

# =========================
# SAVE MODEL
# =========================
os.makedirs("models", exist_ok=True)
dqn_model.save("models/cityflow_dqn.keras")

# =========================
# TRAINING SUMMARY
# =========================
end_time = time.time()
training_time = end_time - start_time

print("\n=== TRAINING SUMMARY ===")
print(f"Episodes: {EPISODES}")
print(f"Time: {training_time:.2f} sec")
print(f"        ({training_time/60:.2f} min)")

# =========================
# PLOTS
# =========================
os.makedirs("plots", exist_ok=True)

# episode reward trend
plt.figure()
plt.plot(episode_rewards)
plt.title("Episode Reward Trend")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.grid()
plt.savefig("plots/episode_rewards.png", dpi=200)
plt.close()

# best episode dynamics
if len(best_step_history) > 0:

    fig, axes = plt.subplots(1, 2, figsize=(14,5))

    axes[0].plot(best_step_history, best_reward_history)
    axes[0].set_title("Cumulative Reward (Best Episode)")
    axes[0].grid()

    axes[1].plot(best_step_history, best_queue_history)
    axes[1].set_title("Queue Length (Best Episode)")
    axes[1].grid()

    plt.tight_layout()
    plt.savefig("plots/best_episode.png", dpi=200)
    plt.close()

print("\n✅ Training complete.")