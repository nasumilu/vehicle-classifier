from vehicle_classifier import VehicleClassifier

servers = ['192.168.1.204:9092']


def main():
    classifier = VehicleClassifier(kafka_server=servers)
    classifier.run()


if __name__ == '__main__':
    main()
