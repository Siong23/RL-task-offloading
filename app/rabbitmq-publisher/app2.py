import pika
import requests
import time

# --- Configuration ---
rabbitmq_host = "100.71.114.85"
target_instance = "100.101.249.101:9100"  # CHANGE to the instance (IP:port) of your desired node
prometheus_url = "http://100.71.114.85:9090"
load_threshold = 50.0
check_interval = 30
message_sent = False

# Prometheus query for a specific node instance
prometheus_query = f'100 - (avg by(instance)(rate(node_cpu_seconds_total{{mode="idle",instance="{target_instance}"}}[1m])) * 100)'

# --- Helper Function ---
def get_average_cpu_load():
    try:
        response = requests.get(f"{prometheus_url}/api/v1/query", params={'query': prometheus_query})
        result = response.json()
        if result['status'] == 'success' and result['data']['result']:
            return float(result['data']['result'][0]['value'][1])
    except Exception as e:
        print(f"Error querying Prometheus: {e}")
    return None

# --- Main Monitoring Loop ---
while True:
    cpu_load = get_average_cpu_load()
    if cpu_load is not None:
        print(f"[INFO] CPU load for {target_instance}: {cpu_load:.2f}%")

        if cpu_load < load_threshold and not message_sent:
            print("[INFO] CPU load is low. Sending message to start AI task.")

            try:
                credentials = pika.PlainCredentials("user", "password")
                connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, credentials=credentials))
                channel = connection.channel()

                channel.queue_declare(queue="ai-task")
                channel.basic_publish(exchange="", routing_key="ai-task", body="Start AI task on Raspberry Pi")

                connection.close()
                message_sent = True
                print("[INFO] Message sent successfully.")
            except Exception as e:
                print(f"[ERROR] Failed to send message: {e}")
        elif cpu_load >= load_threshold:
            message_sent = False  # Reset trigger

    time.sleep(check_interval)
