import json
import base64
import logging
from threading import Thread
from kafka import KafkaConsumer, KafkaProducer
from kafka.errors import KafkaError


class Consumer(object):
    """
        Class used to consume and decode the Kafka 'motion' topic messages and apply a callback on a different
        thread
    """
    __consumer__ = None

    def __init__(self, servers: list = None, topic: str = 'motion', group_id: str = 'spark'):
        """ Initialize Kafa Consumer """

        if servers is None:
            servers = ['localhost:9092']

        self.__consumer__ = KafkaConsumer(
            topic,
            group_id=group_id,
            bootstrap_servers=servers
        )

    def consume(self, callback):
        """
            The consumer which consumes and decodes the kafka message and applies a callback on a different
            thread
        """
        for message in self.__consumer__:
            data = json.loads(message.value.decode())  # decode the data
            data['video'] = base64.b64decode(data['video'])  # decode the bas64 video
            thread = Thread(target=callback, args=[data])  # init the new thread with the decoded data
            thread.start()  # invoke the new thread


class Producer(object):
    __topic__ = None
    __producer__ = None
    __classifier__ = None

    def __init__(self, servers: list = None, topic: str = 'vehicle_prediction'):
        """ Initialize Kafa Producer """
        if servers is None:
            servers = ['localhost:9092']

        self.__producer__ = KafkaProducer(
            bootstrap_servers=servers,
            value_serializer=lambda m: json.dumps(m).encode()
        )
        self.__topic__ = topic

    def publish(self, data):
        promise = self.__producer__.send(self.__topic__, value=data)
        try:
            metadata = promise.get(timeout=10)
        except KafkaError:
            logging.exception()
            pass

        self.__producer__.flush()
