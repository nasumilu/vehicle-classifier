import base64
import math

import tensorflow as tf
import numpy as np
import cv2

from messenger import Producer, Consumer


class VehicleClassifier(object):
    """ Used to classify a vehicle using a preconfigured model """
    __save_path__: str = None
    __consumer__: Consumer = None
    __producer__: Producer = None
    __model_path__: str = None
    __dimensions__: tuple = None
    __categories__: list = None
    __model__ = None

    def __classify__(self, image) -> dict:
        """ Classifies a single image """
        img_array = tf.keras.utils.array_to_img(image)
        img_array = tf.expand_dims(img_array, 0)
        predictions = self.__model__.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        return {
            'prediction': self.__categories__[int(np.argmax(score))],
            'confidence': np.max(score) * 100
        }

    def classifyVideo(self, source) -> list:
        """ Classifies a video by extracting and classifying each frame """
        video = cv2.VideoCapture(source)
        success, image = video.read()
        count = 0
        predictions = []
        while success:
            img = cv2.resize(image, self.__dimensions__)
            prediction = self.__classify__(img)
            prediction['frame'] = count
            prediction['image'] = base64.b64encode(cv2.imencode('.jpg', image)[1].tobytes()).decode()
            predictions.append(prediction)
            success, image = video.read()
            count += 1
        video.release()
        return predictions

    def run(self) -> str:
        self.__consumer__.consume(self.__handleMessage__)

    def __save_video__(self, data):
        filename = f"{self.__save_path__}/{data['event_datetime']}.{data['event']}.mkv"
        with(open(filename, 'wb') as image):
            image.write(data['video'])
            image.close()
        return filename

    @staticmethod
    def __median_frame__(predictions):
        size = len(predictions) / 2
        # odd number of frames send the true median
        if size % 2 == 0:
            return predictions[int(size)]
        # even number of frames send the median frame with the highest confidence
        return max([
            predictions[math.floor(size)],
            predictions[math.ceil(size)]],
            key=lambda prediction: prediction['confidence']
        )

    def __handleMessage__(self, data):
        filename = self.__save_video__(data)
        del data['video']
        del data['file']
        predictions = self.classifyVideo(filename)
        data['prediction'] = self.__median_frame__(predictions)
        self.__producer__.publish(data)

    def __init__(self,
                 kafka_server: list = None,
                 model_path: str = None,
                 dimensions: str = None,
                 categories: list = None,
                 save_path: str = None):

        if save_path is None:
            save_path = './uploads'

        if kafka_server is None:
            kafka_server = ['localhost:9092']

        # default model path value
        if model_path is None:
            model_path = './model'

        # default image dimension value
        if dimensions is None:
            dimensions = (int(640 / 4), int(480 / 4))

        # default categories
        if categories is None:
            categories = ['PASSENGER_VEHICLE', 'TRUCK_VAN']

        self.__consumer__ = Consumer(servers=kafka_server)
        self.__producer__ = Producer(servers=kafka_server)
        self.__model__ = tf.keras.models.load_model(model_path)  # init the model
        self.__dimensions__ = dimensions
        self.__categories__ = categories
        self.__save_path__ = save_path.rstrip('/')  # strip any trailing /
