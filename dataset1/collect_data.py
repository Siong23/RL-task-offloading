import requests
import urllib.parse
import time
import csv
import os

# IP Addresses
IOT_IP = "100.123.199.63"
EDGE_IP = "100.106.97.77"

# Prometheus configuration
PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

# Create dataset folder if it doesn't exist
os.makedirs("dataset", exist_ok=True)

OUTPUT_FILE = "dataset/training_dataset.csv"

# Define Prometheus CPU queries for each device
def get_query(ip):
    return f'''(1 - sum without (mode) (irate(node_cpu_seconds_total{{job="node-exporter", mode=~"idle|iowait|steal", instance="{ip}:9100", cluster=""}}[2m]))) / ignoring(cpu) group_left count without (cpu, mode) (node_cpu_seconds_total{{job="node-exporter", mode="idle", instance="{ip}:9100", cluster=""}})'''

QUERIES = {
    "iot": get_query(IOT_IP),
    "edge": get_query(EDGE_IP)
}

# Function to query Prometheus
def query_prometheus(query):
    encoded_query = urllib.parse.quote(query, safe="()[],")
    full_url = f"{PROMETHEUS_URL}?query={encoded_query}"
    try:
        response = requests.get(full_url)
        data = response.json()
        if 'data' in data and 'result' in data['data'] and len(data['data']['result']) > 0:
            return float(data['data']['result'][0]['value'][1]) * 100  # Convert to %
        else:
            return 0.0
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
        return 0.0

# Collect and save data for a specific time interval
def collect_data(interval_seconds, duration_seconds, writer):
    timeslots = duration_seconds // interval_seconds
    for slot in range(1, int(timeslots) + 1):
        edge_cpu = query_prometheus(QUERIES["edge"])
        iot_cpu = query_prometheus(QUERIES["iot"])
        writer.writerow([f"{interval_seconds}s_{slot}", round(edge_cpu, 2), round(iot_cpu, 2)])
        print(f"[{interval_seconds}s Slot {slot}] Edge={edge_cpu:.2f}% | IoT={iot_cpu:.2f}%")
        time.sleep(interval_seconds)

if __name__ == "__main__":
    try:
        # Duration of each sampling session (in seconds)
        DURATION = 300  # 5 minutes

        print(f"\nCollecting Prometheus CPU data → {OUTPUT_FILE}")

        with open(OUTPUT_FILE, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["time slot", "edge cpu data", "iot cpu data"])

            # Collect data for different intervals
            collect_data(30, DURATION, writer)
            collect_data(60, DURATION, writer)
            collect_data(300, DURATION, writer)

        print(f"\n✅ All data saved to: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print("\nStopped by user.")
