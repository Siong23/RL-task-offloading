import requests
import urllib.parse
import time
import csv
import os

# Node IP addresses
NODES = {
    "iot1": "192.168.0.160",
    "edge1": "192.168.0.147",
    "iot2": "192.168.0.159",
    "edge2": "192.168.0.124"
}

# Prometheus configuration
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)
OUTPUT_FILE = "dataset/training_dataset_rnd_stress_7.csv"

# PromQL templates
def get_queries(ip):
    return {
        "cpu": f'''100 * (1 - avg by (instance) (
            irate(node_cpu_seconds_total{{
                job="node-exporter",
                mode=~"idle|iowait|steal",
                instance="{ip}:9100"
            }}[1m])
        ))''',
        "memory": f'''100 * (1 - (
            node_memory_MemAvailable_bytes{{instance="{ip}:9100"}} /
            node_memory_MemTotal_bytes{{instance="{ip}:9100"}}
        ))''',
        "disk_io": f'''sum by (instance) (
            irate(node_disk_read_bytes_total{{instance="{ip}:9100"}}[1m]) +
            irate(node_disk_written_bytes_total{{instance="{ip}:9100"}}[1m])
        )''',
        "net_rx": f'''sum by (instance) (
            irate(node_network_receive_bytes_total{{instance="{ip}:9100", device!~"lo"}}[1m])
        )''',
        "net_tx": f'''sum by (instance) (
            irate(node_network_transmit_bytes_total{{instance="{ip}:9100", device!~"lo"}}[1m])
        )'''
    }

# Function to query Prometheus
def query_prometheus(query):
    encoded_query = urllib.parse.quote(query, safe="()[],=*\"{} ")
    full_url = f"{PROMETHEUS_URL}?query={encoded_query}"
    try:
        response = requests.get(full_url)
        data = response.json()
        if "data" in data and "result" in data["data"] and len(data["data"]["result"]) > 0:
            return float(data["data"]["result"][0]["value"][1])
        else:
            return 0.0
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return 0.0

# Collect and save data for a specific time interval
def collect_data(interval_seconds, duration_seconds, writer):
    timeslots = duration_seconds // interval_seconds
    for slot in range(1, int(timeslots) + 1):
        row = [f"{interval_seconds}s_{slot}"]
        print(f"\n[{interval_seconds}s Slot {slot}]")
        for name, ip in NODES.items():
            q = get_queries(ip)
            cpu = query_prometheus(q["cpu"])
            mem = query_prometheus(q["memory"])
            disk = query_prometheus(q["disk_io"])
            net_rx = query_prometheus(q["net_rx"])
            net_tx = query_prometheus(q["net_tx"])
            row.extend([round(cpu, 2), round(mem, 2), round(disk, 2), round(net_rx, 2), round(net_tx, 2)])
            print(f"  {name}: CPU={cpu:.2f}% | MEM={mem:.2f}% | DISK={disk:.2f} B/s | RX={net_rx:.2f} B/s | TX={net_tx:.2f} B/s")
        writer.writerow(row)
        time.sleep(interval_seconds)

if __name__ == "__main__":
    try:
        DURATION = 4900  # seconds

        print(f"\nCollecting Prometheus metrics → {OUTPUT_FILE}")

        with open(OUTPUT_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)

            # CSV header
            header = ["time slot"]
            for name in NODES.keys():
                header.extend([
                    f"{name}_cpu(%)", f"{name}_mem(%)",
                    f"{name}_disk_io(B/s)", f"{name}_net_rx(B/s)", f"{name}_net_tx(B/s)"
                ])
            writer.writerow(header)

            # Collect data every 5 seconds
            collect_data(5, DURATION, writer)

        print(f"\n✅ All data saved to: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\n⏹️ Stopped by user.")
