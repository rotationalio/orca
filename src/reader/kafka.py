from src.reader.reader import Reader

import json

from kafka import KafkaConsumer

class TopicReader(Reader):
    """
    TopicReader implements reading data from a Kafka topic into a stream.
    """
    def __init__(self, url, topic):
        self.consumer = KafkaConsumer(topic, bootstrap_servers=url)

    def read(self):
        """
        Read data from the Kafka topic into a stream.
        """
        for message in self.consumer:
            try:
                yield json.loads(message.value)
            except Exception as e:
                print("Error parsing topic: " + str(e))