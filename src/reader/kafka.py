from src.reader.reader import Reader

from kafka import KafkaConsumer
from kafka.structs import OffsetAndMetadata, TopicPartition

class TopicReader(Reader):
    """
    TopicReader implements reading data from a Kafka topic into a stream.
    Parameters:
        url:
            URL that points to a Kafka broker.
        topic:
            Kafka topic name to subscribe to.
        resume_earliest:
            If True, the consumer will start reading from the last committed offset
            after a restart. If False, the consumer will start reading at the end of
            the partition log. Default: True
        auto_commit:
            If True, the consumer will automatically commit read offsets every interval.
            If False, the caller is responsible for committing offsets with the
            commit() method. Default: True
        group_id:
            Kafka consumer group id, this is used internally by the consumer as an
            identifier for automatic offset committing. Default: 'orca'
        auto_commit_interval:
            Time in milliseconds between automatic offset commits. Default: 5000
    """
    def __init__(self, url, topic, resume_earliest=True, auto_commit=True, group_id="orca", auto_commit_interval=5000):
        if not isinstance(url, str) or url == "":
            raise ValueError("Kafka broker URL is required.")
        if not isinstance(topic, str) or topic == "":
            raise ValueError("Kafka topic is required.")
        self.topic = topic
        if resume_earliest:
            self.auto_offset_reset = "earliest"
        else:
            self.auto_offset_reset = "latest"
        if auto_commit and group_id == "":
            raise ValueError("Kafka group_id is required if auto_commit is True.")
        self.message_ = None
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=url,
                auto_offset_reset=self.auto_offset_reset,
                enable_auto_commit=auto_commit,
                group_id=group_id,
                auto_commit_interval_ms=auto_commit_interval)
        except Exception as e:
            raise ValueError("Error configuring Kafka consumer: " + str(e))

    def read(self):
        """
        Read data from the Kafka topic into a stream.
        """
        for message in self.consumer:
            self.message_ = message
            yield message.value

    def commit(self):
        """
        Commit the last read offset to Kafka. This enables the caller to control when
        messages are considered 'committed' based on external logic.
        """
        if self.message_ is None:
            raise ValueError("No message to commit.")
        self.consumer.commit({TopicPartition(self.topic, self.message_.partition): OffsetAndMetadata(self.message_.offset + 1, None)})