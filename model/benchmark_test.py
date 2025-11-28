import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import pickle

np.random.seed(42)

H = 124500  # Task size in bits
C_iot = 7.2e9
C_edge = 13.2e9
phi = 6650
E1 = 0.1
BETA = 0.5
T = 0.5

CPU_THRESHOLD = 0.3

class BenchmarkEnvironment:
    def __init__(self, benchmark_data):
        self.benchmark_data = benchmark_data
        self.episode_count = 0
        self.reset()

    def reset(self):
        self.step_count = 0
        self.prev_action = 0
        self.episode_start_idx = (self.episode_count * 100) % max(len(self.benchmark_data), 1)

        if len(self.benchmark_data) > 0:
            first_row = self.benchmark_data.iloc[self.episode_start_idx]
            self.edge_cpu_usage = first_row['edge_cpu']
            self.iot_cpu_usage = first_row['iot_cpu']
        else:
            self.edge_cpu_usage = 0.05
            self.iot_cpu_usage = 0.05

        self.episode_count += 1
        return self.get_state()

    def get_state(self):
        return (self.iot_cpu_usage, self.edge_cpu_usage, self.prev_action)

    def update_cpu_usage(self, action):
        if len(self.benchmark_data) > 0:
            row_idx = self.episode_start_idx + self.step_count
            row_idx = min(row_idx, len(self.benchmark_data) - 1)
            current_row = self.benchmark_data.iloc[row_idx]
            base_iot_cpu = current_row['iot_cpu']
            base_edge_cpu = current_row['edge_cpu']
        else:
            base_iot_cpu = 0.05
            base_edge_cpu = 0.05

        iot_task_load = (1 - action) * (phi * H) / (C_iot * T)
        edge_task_load = action * (phi * H) / (C_edge * T)

        self.iot_cpu_usage = np.clip(base_iot_cpu + iot_task_load, 0.05, 0.99)
        self.edge_cpu_usage = np.clip(base_edge_cpu + edge_task_load, 0.05, 0.99)

    def step(self, action):
        self.update_cpu_usage(action)

        epsilon = 1e-6
        available_iot_capacity = max(C_iot * (1 - self.iot_cpu_usage), epsilon)
        available_edge_capacity = max(C_edge * (1 - self.edge_cpu_usage), epsilon)

        L_et_raw = (2 * action - 1) * (phi * H) / (available_edge_capacity * T) + self.edge_cpu_usage
        L_dt_raw = (2 * (1 - action) - 1) * (phi * H) / (available_iot_capacity * T) + self.iot_cpu_usage

        L_et = np.clip(L_et_raw, 0, 1)
        L_dt = np.clip(L_dt_raw, 0, 1)

        capacity_violation = 1 if (L_et >= 1 and action == 1) or (L_dt >= 1 and action == 0) else 0

        C_1t = min(np.abs(L_dt - L_et), 1e6)
        C_2t = int(action != self.prev_action)

        total_cost = BETA * C_1t + (1 - BETA) * C_2t
        reward = -total_cost - E1 * capacity_violation

        self.prev_action = action
        self.step_count += 1

        return self.get_state(), reward, C_1t, C_2t, total_cost, capacity_violation

class QLearningMethod:
    def __init__(self, model_path):
        try:
            with open(model_path, 'rb') as f:
                self.q_table = defaultdict(lambda: np.zeros(2), pickle.load(f))
        except FileNotFoundError:
            print(f"Q-Learning model file '{model_path}' not found. Using random initialization.")
            self.q_table = defaultdict(lambda: np.random.random(2))

    def get_action(self, state):
        iot_cpu, edge_cpu, prev_action = state
        iot_cpu_level = min(int(iot_cpu * 10), 9)
        edge_cpu_level = min(int(edge_cpu * 10), 9)
        return np.argmax(self.q_table[(iot_cpu_level, edge_cpu_level, prev_action)])

class StaticThresholdMethod:
    def get_action(self, state):
        iot_cpu, _, _ = state
        return 1 if iot_cpu > CPU_THRESHOLD else 0

class RandomMethod:
    def get_action(self, state):
        return np.random.randint(0, 2)

class GreedyMethod:
    def get_action(self, state):
        iot_cpu, edge_cpu, _ = state
        return 1 if edge_cpu < iot_cpu else 0

def evaluate_method(method, benchmark_data, episodes=50, steps_per_episode=100):
    env = BenchmarkEnvironment(benchmark_data)

    rewards, costs, ratios, iot_cpus, edge_cpus = [], [], [], [], []
    c1t_histories, c2t_histories = [], []

    for episode in range(episodes):
        state = env.reset()
        total_reward, total_cost, offloads = 0, 0, 0
        iot_cpu_list, edge_cpu_list = [], []
        c1t_list, c2t_list = [], []

        for step in range(steps_per_episode):
            action = method.get_action(state)
            state, reward, c1t, c2t, cost, _ = env.step(action)

            total_reward += reward
            total_cost += cost
            offloads += action
            iot_cpu_list.append(env.iot_cpu_usage)
            edge_cpu_list.append(env.edge_cpu_usage)
            c1t_list.append(c1t)
            c2t_list.append(c2t)

        rewards.append(total_reward)
        costs.append(total_cost)
        ratios.append(offloads / steps_per_episode)
        iot_cpus.append(np.mean(iot_cpu_list))
        edge_cpus.append(np.mean(edge_cpu_list))
        c1t_histories.append(np.mean(c1t_list))
        c2t_histories.append(np.mean(c2t_list))

    return {
        'rewards': np.mean(rewards),
        'costs': np.mean(costs),
        'offload_ratio': np.mean(ratios),
        'iot_cpu': np.mean(iot_cpus),
        'edge_cpu': np.mean(edge_cpus),
        'reward_history': rewards,
        'cost_history': costs,
        'offload_history': ratios,
        'c1t_history': c1t_histories,
        'c2t_history': c2t_histories
    }

def plot_comparison(results):
    methods = list(results.keys())
    episodes = len(results[methods[0]]['cost_history'])

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    method_colors = {method: colors[i % len(colors)] for i, method in enumerate(methods)}

    # Plot 1: C1t per Episode
    for m in methods:
        axs[0, 0].plot(range(1, episodes + 1), results[m]['c1t_history'],
                      label=m, color=method_colors[m], linewidth=2, marker='x', markersize=3)
    axs[0, 0].set_title('C1t per Episode', fontsize=14, fontweight='bold')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('C1t Value')
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].grid(True, alpha=0.3)

    # Plot 2: C2t per Episode
    for m in methods:
        axs[0, 1].plot(range(1, episodes + 1), results[m]['c2t_history'],
                      label=m, color=method_colors[m], linewidth=2, marker='D', markersize=3)
    axs[0, 1].set_title('C2t per Episode', fontsize=14, fontweight='bold')
    axs[0, 1].set_xlabel('Episode')
    axs[0, 1].set_ylabel('C2t Value')
    axs[0, 1].legend(fontsize=10)
    axs[0, 1].grid(True, alpha=0.3)

    # Plot 3: Total Cost per Episode
    for m in methods:
        axs[1, 0].plot(range(1, episodes + 1), results[m]['cost_history'],
                      label=m, color=method_colors[m], linewidth=2, marker='s', markersize=3)
    axs[1, 0].set_title('Total Cost per Episode', fontsize=14, fontweight='bold')
    axs[1, 0].set_xlabel('Episode')
    axs[1, 0].set_ylabel('Total Cost')
    axs[1, 0].legend(fontsize=10)
    axs[1, 0].grid(True, alpha=0.3)

    # Plot 4: Average CPU Usage (Bar Chart)
    x = np.arange(len(methods))
    width = 0.35
    iot_means = [results[m]['iot_cpu'] for m in methods]
    edge_means = [results[m]['edge_cpu'] for m in methods]

    bars1 = axs[1, 1].bar(x - width/2, iot_means, width, label='IoT CPU',
                         color='lightblue', edgecolor='black', linewidth=1)
    bars2 = axs[1, 1].bar(x + width/2, edge_means, width, label='Edge CPU',
                         color='lightcoral', edgecolor='black', linewidth=1)

    for bar in bars1:
        height = bar.get_height()
        axs[1, 1].annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        axs[1, 1].annotate(f'{height:.2f}',
                          xy=(bar.get_x() + bar.get_width() / 2, height),
                          xytext=(0, 3),
                          textcoords="offset points",
                          ha='center', va='bottom', fontsize=9)

    axs[1, 1].set_title('Average CPU Usage', fontsize=14, fontweight='bold')
    axs[1, 1].set_xlabel('Method')
    axs[1, 1].set_ylabel('CPU Usage')
    axs[1, 1].set_xticks(x)
    axs[1, 1].set_xticklabels(methods, rotation=45, ha='right')
    axs[1, 1].legend(fontsize=10)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim(0, 1)

    plt.tight_layout()
    plt.show()

def load_benchmark_data(filename):
    try:
        df = pd.read_csv(filename)
        expected_columns = ['time_slot', 'edge_cpu', 'iot_cpu']
        if not all(col in df.columns for col in expected_columns):
            column_mapping = {'time slot': 'time_slot', 'edge cpu': 'edge_cpu', 'iot cpu': 'iot_cpu'}
            df = df.rename(columns=column_mapping)
        print(f"Loaded {len(df)} rows of benchmark data")
        return df
    except FileNotFoundError:
        print(f"Benchmark file '{filename}' not found. Using default values.")
        return pd.DataFrame()

def main():
    benchmark_data = load_benchmark_data('training_data.csv')

    methods = {
        'Q-Learning': QLearningMethod('simulation_dataset_q_learning_model.pkl'),
        'Static Threshold': StaticThresholdMethod(),
        'Random': RandomMethod(),
        'Greedy': GreedyMethod()
    }

    print("Evaluating methods...")
    results = {}
    for name, method in methods.items():
        print(f"Evaluating {name}...")
        results[name] = evaluate_method(method, benchmark_data)

    print("\nComparison Results:")
    print("-" * 95)
    print(f"|{'Method':<18}| {'Cost':>12} | {'C1t':>12} | {'C2t':>12} | {'IoT CPU %':>12} | {'Edge CPU %':>12} |")
    print("-" * 95)
    for name, res in results.items():
        print(f"|{name:<18}| {res['costs']:>12.2f} | {np.mean(res['c1t_history']):>12.2f} | {np.mean(res['c2t_history']):>12.2f} | "
              f"{res['iot_cpu']*100:>11.1f}% | {res['edge_cpu']*100:>11.1f}% |")
    print("-" * 95)

    print("\nAdditional Statistics:")
    for name, res in results.items():
        print(f"\n{name}:")
        print(f"  Reward std: {np.std(res['reward_history']):.2f}")
        print(f"  Cost std: {np.std(res['cost_history']):.2f}")
        print(f"  C1t std: {np.std(res['c1t_history']):.2f}")
        print(f"  C2t std: {np.std(res['c2t_history']):.2f}")

    plot_comparison(results)

if __name__ == "__main__":
    main()
