import pika

rabbitmq_host = "rabbitmq"
credentials = pika.PlainCredentials("user", "password")
connection = pika.BlockingConnection(pika.ConnectionParameters(host=rabbitmq_host, credentials=credentials))
channel = connection.channel()

channel.queue_declare(queue="ai-task")
channel.basic_publish(exchange="", routing_key="ai-task", body="Start AI task on Raspberry Pi")

connection.close()
