import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

np.random.seed(42)

# Constants
H = 124500  # Task size in bits
C_iot = 7.2e9  # IoT capacity in CPU cycles per second
C_edge = 13.2e9  # Edge server capacity in CPU cycles per second
phi = 6650  # Computation intensity in CPU cycles per bit
E1 = 0.1  # Penalty for capacity violation
LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.9
EPISODES = 150
NUM_IOT_DEVICES = 1  # Fixed to 1
BETA = 0.5  # Weight for balancing C_1t and C_2t

T = 0.5  # Edge Time slot duration in seconds

def discretize_state(iot_cpu, edge_cpu, prev_action):
    iot_cpu_level = min(int(iot_cpu * 10), 9)
    edge_cpu_level = min(int(edge_cpu * 10), 9)
    return (iot_cpu_level, edge_cpu_level, prev_action)

class Environment:
    def __init__(self, training_data):
        self.training_data = training_data
        self.episode_count = 0
        self.reset()

    def reset(self):
        self.state = 0
        self.prev_action = 0
        self.step_count = 0
        self.episode_start_idx = (self.episode_count * 10000) % len(self.training_data)

        if len(self.training_data) > 0:
            first_row = self.training_data.iloc[self.episode_start_idx]
            self.edge_cpu_usage = first_row['edge_cpu']
            self.iot_cpu_usage = first_row['iot_cpu']
        else:
            self.edge_cpu_usage = 0.05
            self.iot_cpu_usage = 0.05

        return discretize_state(self.iot_cpu_usage, self.edge_cpu_usage, self.prev_action)

    def update_cpu_from_dataset(self):
        if len(self.training_data) > 0:
            row_idx = self.episode_start_idx + self.step_count
            row_idx = min(row_idx, len(self.training_data) - 1)
            current_row = self.training_data.iloc[row_idx]
            return current_row['edge_cpu'], current_row['iot_cpu']
        else:
            return 0.05, 0.05

    def step(self, action):
        base_edge_cpu, base_iot_cpu = self.update_cpu_from_dataset()

        iot_task_load = (1 - action) * (phi * H) / (C_iot * T)
        edge_task_load = action * (phi * H) / (C_edge * T)

        self.iot_cpu_usage = np.clip(base_iot_cpu + iot_task_load, 0.05, 0.99)
        self.edge_cpu_usage = np.clip(base_edge_cpu + edge_task_load, 0.05, 0.99)

        epsilon = 1e-6
        available_iot_capacity = max(C_iot * (1 - self.iot_cpu_usage), epsilon)
        available_edge_capacity = max(C_edge * (1 - self.edge_cpu_usage), epsilon)

        L_et_raw = (2 * action - 1) * (phi * H) / (available_edge_capacity * T) + self.edge_cpu_usage
        L_dt_raw = (2 * (1 - action) - 1) * (phi * H) / (available_iot_capacity * T) + self.iot_cpu_usage

        L_et = np.clip(L_et_raw, 0, 1)
        L_dt = np.clip(L_dt_raw, 0, 1)

        if action == 1:
            capacity_violation = 1 if L_et >= 1 else 0
        else:
            capacity_violation = 1 if L_dt >= 1 else 0

        C_1t = min(np.abs(L_dt - L_et), 1e6)
        C_2t = int(action != self.prev_action)

        total_cost = BETA * C_1t + (1 - BETA) * C_2t
        reward = -total_cost - E1 * capacity_violation

        self.prev_action = action
        self.step_count += 1

        if self.step_count >= 10000:
            self.episode_count += 1

        next_state = discretize_state(self.iot_cpu_usage, self.edge_cpu_usage, self.prev_action)
        return (next_state, reward, C_1t, C_2t, total_cost, capacity_violation, self.iot_cpu_usage, self.edge_cpu_usage)

class QLearningAgent:
    def __init__(self):
        self.q_table = defaultdict(lambda: np.zeros(2))

    def get_action(self, state, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(0, 2)
        else:
            return np.argmax(self.q_table[state])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state][action]
        next_max_q = np.max(self.q_table[next_state])
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * next_max_q)
        delta = abs(new_q - current_q)
        self.q_table[state][action] = new_q
        return delta

    def export_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.q_table), f)

    @classmethod
    def import_model(cls, filename):
        with open(filename, 'rb') as f:
            q_table = defaultdict(lambda: np.zeros(2), pickle.load(f))
        agent = cls()
        agent.q_table = q_table
        return agent

def load_training_data(filename):
    try:
        df = pd.read_csv(filename)
        expected_columns = ['time_slot', 'edge_cpu', 'iot_cpu']
        if not all(col in df.columns for col in expected_columns):
            column_mapping = {
                'time slot': 'time_slot',
                'edge cpu': 'edge_cpu',
                'iot cpu': 'iot_cpu'
            }
            df = df.rename(columns=column_mapping)
        print(f"Loaded {len(df)} rows of training data")
        return df
    except FileNotFoundError:
        print(f"Training data file '{filename}' not found. Using default values.")
        return pd.DataFrame()

def run_simulation():
    training_data = load_training_data('training_data.csv')
    env = Environment(training_data)
    agent = QLearningAgent()

    rewards_per_episode = []
    cost_1_per_episode = []
    cost_2_per_episode = []
    total_cost_per_episode = []
    offload_ratio_per_episode = []
    q_value_changes_per_episode = []
    cumulative_rewards = []
    eps_start = 1.0
    eps_end = 0.05
    TRAIN_EPISODES = 100
    DECAY_FACTOR = (eps_start - eps_end) / TRAIN_EPISODES
    eps = eps_start

    for episode in range(EPISODES):
        state = env.reset()
        total_reward = 0
        total_cost_1 = 0
        total_cost_2 = 0
        total_cost = 0
        decisions = []
        capacity_violations = []
        q_deltas = []

        for step in range(10000):
            action = agent.get_action(state, epsilon=eps)
            next_state, reward, cost_1, cost_2, step_total_cost, capacity_violation, iot_cpu, edge_cpu = env.step(action)

            delta = agent.update_q_table(state, action, reward, next_state)
            q_deltas.append(delta)

            total_reward += reward
            total_cost_1 += cost_1
            total_cost_2 += cost_2
            total_cost += step_total_cost

            decisions.append(action)
            capacity_violations.append(capacity_violation)

            state = next_state

        eps = max(eps - DECAY_FACTOR, eps_end)
        rewards_per_episode.append(total_reward)
        cost_1_per_episode.append(total_cost_1)
        cost_2_per_episode.append(total_cost_2)
        total_cost_per_episode.append(total_cost)

        offload_ratio = sum(decisions) / len(decisions)
        offload_ratio_per_episode.append(offload_ratio)
        cumulative_rewards.append(np.sum(rewards_per_episode))
        q_value_changes_per_episode.append(np.sum(q_deltas))

        print(f"Episode {episode + 1}:")
        print(f"Total offloading decisions: {sum(decisions)}")
        print(f"Total capacity violations: {sum(capacity_violations):.4f}")
        print(f"Total reward: {total_reward:.4f}")
        print(f"Total cost 1: {total_cost_1:.4f}")
        print(f"Total cost 2: {total_cost_2:.4f}")
        print(f"Total cost: {total_cost:.4f}")
        print(f"Offloading ratio: {offload_ratio:.4f}")
        print("---")

    fig, axs = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Q-Learning Training Results')

    axs[0, 0].plot(rewards_per_episode, 'b-')
    axs[0, 0].set_title('Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Value')
    axs[0, 0].grid(True)

    axs[0, 1].plot(cost_1_per_episode, 'r-')
    axs[0, 1].set_title('Cost 1 (Load Imbalance)')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('Value')
    axs[0, 1].grid(True)

    axs[0, 2].plot(cost_2_per_episode, 'g-')
    axs[0, 2].set_title('Cost 2 (Switching)')
    axs[0, 2].set_xlabel('Episode')
    axs[0, 2].set_ylabel('Value')
    axs[0, 2].grid(True)

    axs[1, 0].plot(total_cost_per_episode, 'm-')
    axs[1, 0].set_title('Total Cost')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Value')
    axs[1, 0].grid(True)

    axs[1, 1].plot(offload_ratio_per_episode, 'c-')
    axs[1, 1].set_title('Offloading Ratio')
    axs[1, 1].set_xlabel('Episode')
    axs[1, 1].set_ylabel('Ratio')
    axs[1, 1].set_ylim(0, 1)
    axs[1, 1].grid(True)

    window_size = 10
    if len(rewards_per_episode) >= window_size:
        moving_avg = np.convolve(rewards_per_episode, np.ones(window_size)/window_size, mode='valid')
        axs[1, 2].plot(range(window_size-1, len(rewards_per_episode)), moving_avg, 'purple')
        axs[1, 2].set_title(f'Learning Progress (Moving Avg, window={window_size})')
        axs[1, 2].set_xlabel('Episode')
        axs[1, 2].set_ylabel('Average Reward')
        axs[1, 2].grid(True)

        axs[1, 2].plot(range(window_size - 1, len(rewards_per_episode)), moving_avg, 'purple')
    axs[1, 2].set_title(f'Learning Progress (Moving Avg, window={window_size})')
    axs[1, 2].set_xlabel('Episode')
    axs[1, 2].set_ylabel('Average Reward')
    axs[1, 2].grid(True)

    fig2, axs2 = plt.subplots(1, 2, figsize=(14, 5))

    axs2[0].plot(cumulative_rewards, color='blue')
    axs2[0].set_title('Cumulative Reward Over Episodes')
    axs2[0].set_xlabel('Episode')
    axs2[0].set_ylabel('Cumulative Reward')
    axs2[0].grid(True)

    axs2[1].plot(q_value_changes_per_episode, color='orange')
    axs2[1].set_title('Q-Value Change Magnitude per Episode')
    axs2[1].set_xlabel('Episode')
    axs2[1].set_ylabel('Q-value Δ Sum')
    axs2[1].grid(True)

    plt.tight_layout()
    plt.savefig('q_learning_analysis_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

    agent.export_model('simulation_dataset_q_learning_model.pkl')
    print("Model exported to simulation_dataset_q_learning_model.pkl")

    print("\n=== Final Training Statistics ===")
    print(f"Total episodes: {EPISODES}")
    print(f"Final reward: {rewards_per_episode[-1]:.4f}")
    print(f"Average reward (last 10 episodes): {np.mean(rewards_per_episode[-10:]):.4f}")
    print(f"Final offloading ratio: {offload_ratio_per_episode[-1]:.4f}")
    print(f"Average offloading ratio: {np.mean(offload_ratio_per_episode):.4f}")

if __name__ == "__main__":
    run_simulation()
