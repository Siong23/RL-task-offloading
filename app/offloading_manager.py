import numpy as np
import pickle
from collections import defaultdict
from kubernetes import client, config
import requests
import urllib.parse
import time

# Prometheus client
from prometheus_client import Gauge, start_http_server

# Constants
H = 124500  # Task size in bits
C_iot = 7.2e9  # IoT capacity in CPU cycles per second
C_edge = 13.2e9  # Edge server capacity in CPU cycles per second
phi = 6650
T = 0.5  # Time slice in seconds 
BETA = 0.9999

# IP Addresses
IOT_IP = "192.168.0.160"
EDGE_IP = "192.168.0.147"

# Prometheus configuration
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"
PROMETHEUS_QUERIES = {
    "iot_device": f'''(1 - sum without (mode) (irate(node_cpu_seconds_total{{job="node-exporter", mode=~"idle|iowait|steal", instance="{IOT_IP}:9100", cluster=""}}[2m]))) / ignoring(cpu) group_left count without (cpu, mode) (node_cpu_seconds_total{{job="node-exporter", mode="idle", instance="{IOT_IP}:9100", cluster=""}})''',
    "edge_device": f'''(1 - sum without (mode) (irate(node_cpu_seconds_total{{job="node-exporter", mode=~"idle|iowait|steal", instance="{EDGE_IP}:9100", cluster=""}}[2m]))) / ignoring(cpu) group_left count without (cpu, mode) (node_cpu_seconds_total{{job="node-exporter", mode="idle", instance="{EDGE_IP}:9100", cluster=""}})'''
}

# --- Prometheus Metrics ---
METRIC_TOTAL_COST = Gauge('myapp_total_cost', 'Combined latency and switching cost')
METRIC_ACTION = Gauge('myapp_offload_action', '1 if offloaded to edge, 0 if local')
METRIC_IOT_CPU = Gauge('myapp_iot_cpu_usage_percent', 'IoT device CPU usage in percent')
METRIC_EDGE_CPU = Gauge('myapp_edge_cpu_usage_percent', 'Edge device CPU usage in percent')

def query_prometheus(query):
    encoded_query = urllib.parse.quote(query, safe="()[],")
    full_url = f"{PROMETHEUS_URL}?query={encoded_query}"
    response = requests.get(full_url)
    data = response.json()
    if 'data' in data and 'result' in data['data']:
        return data['data']['result']
    else:
        return None


def get_cpu_usage():
    iot_cpu_usage = query_prometheus(PROMETHEUS_QUERIES['iot_device'])
    edge_cpu_usage = query_prometheus(PROMETHEUS_QUERIES['edge_device'])

    iot_total_usage = sum(float(metric['value'][1])
                          for metric in iot_cpu_usage)
    edge_total_usage = sum(float(metric['value'][1])
                           for metric in edge_cpu_usage)

    return iot_total_usage, edge_total_usage


def swap_deployment_nodes(decision):
    config.load_kube_config(config_file='/etc/rancher/k3s/k3s.yaml')
    api = client.AppsV1Api()

    deployment_name = "myapp-deployment"
    namespace = "default"

    try:
        deployment = api.read_namespaced_deployment(deployment_name, namespace)
        current_node = deployment.spec.template.spec.node_selector.get("kubernetes.io/hostname")
        new_node = "iot" if decision == 0 else "nuc2"

        if current_node == new_node:
            print(f"No change needed. Current node is already '{current_node}'.\n")
            return

        deployment.spec.template.spec.node_selector = {"kubernetes.io/hostname": new_node}
        api.patch_namespaced_deployment(deployment_name, namespace, deployment)
        print(f"Deployment '{deployment_name}' is being swapped to run on '{new_node}'.")
        wait_for_deployment_to_run(api, deployment_name, namespace)

    except Exception as e:
        print(f"An error occurred: {e}")


def wait_for_deployment_to_run(api, deployment_name, namespace):
    while True:
        status = api.read_namespaced_deployment_status(deployment_name, namespace)
        if status.status.available_replicas and status.status.available_replicas > 0:
            print(f"Deployment '{deployment_name}' is now running with {status.status.available_replicas} replica(s).\n")
            break
        else:
            print(f"Waiting for deployment '{deployment_name}' to be in Running state...\n")
            time.sleep(10)


def discretize_state(iot_cpu, edge_cpu, prev_action):
    iot_cpu_level = min(int(iot_cpu * 10), 9)
    edge_cpu_level = min(int(edge_cpu * 10), 9)
    return (iot_cpu_level, edge_cpu_level, prev_action)


def make_offloading_decision(iot_usage, edge_usage, prev_action, model_path='q_learning_model.pkl'):
    with open(model_path, 'rb') as f:
        q_table = defaultdict(lambda: np.zeros(2), pickle.load(f))
    state = discretize_state(iot_usage, edge_usage, prev_action)
    return np.argmax(q_table[state])


def main():
    # Start Prometheus metrics HTTP server on port 8000
    start_http_server(8000, addr='0.0.0.0')
    prev_action = 0

    while True:
        iot_usage, edge_usage = get_cpu_usage()
        
        iot_percent = iot_usage * 100
        edge_percent = edge_usage * 100
        
        decision = make_offloading_decision(iot_usage, edge_usage, prev_action)
       
        # Raw latency calculations based on decision
        latency_edge_raw = (2 * decision - 1) * (phi * H) / (C_edge * T) + edge_usage
        latency_iot_raw = (2 * (1 - decision) - 1) * (phi * H) / (C_iot * T) + iot_usage

        # Clip the latencies to [0, 1] to bound them
        latency_edge = np.clip(latency_edge_raw, 0, 1)
        latency_iot = np.clip(latency_iot_raw, 0, 1)

        # Cost components
        C1t = min(np.abs(latency_iot - latency_edge), 1e6)  # bounded cost difference
        C2t = int(decision != prev_action)  # switching cost

        C_total = BETA * C1t + (1 - BETA) * C2t

        # Update Prometheus metrics
        METRIC_TOTAL_COST.set(C_total)
        METRIC_ACTION.set(decision)
        METRIC_IOT_CPU.set(iot_percent)
        METRIC_EDGE_CPU.set(edge_percent)

        print(f"Current state: IoT {iot_usage:.2f}, Edge {edge_usage:.2f}")
        print(f"Decision: {'Edge' if decision == 1 else 'IoT'}")

        swap_deployment_nodes(decision)
        prev_action = decision

        time.sleep(60)


if __name__ == "__main__":
    main()
