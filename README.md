# Vehicle Classifier

This is my first every attempt at ML. The project is intended to determine if 
an image contains a passenger vehicle or pickup/cargo-van. This project very closely
follows Tensorflow's Image Classification Tutorial at 
https://www.tensorflow.org/tutorials/images/classification.

## Basic Usage

```shell
$ git clone git@github.com:nasumilu/vehicle-classifier.git
$ cd vehicle-classifier
$ ./dataset
```

It may take some time for the model to compile but once completed the 
directory `model` will be created. This is ths saved model to load and 
classify images. 

Loading and using the model to make a prediction my look something like this:

```python
model = f.keras.models.load_model('./model')
prediction = self.__model__.predict(image_array)
```