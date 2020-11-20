from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import requests
import numpy as np
import os
from flask_cors import CORS
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


app = Flask(__name__)
CORS(app)

PATH = os.path.join(os.getcwd(), 'backend')


def load_model():
    model = keras.models.load_model(os.path.join(PATH, 'entire_model.h5'))
    return model


def transform_image(input_image):
    img_width, img_height = 128, 128
    img = image.load_img(input_image, target_size=(img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    return img


def get_prediction(image):
    model = load_model()
    transformed_img = transform_image(image)
    predictions = model.predict(transformed_img)
    return np.around(predictions[0], decimals=4).tolist()


pred = get_prediction(os.path.join(
    PATH, 'dataset', 'test', 'dogs', 'dog.0.jpg'))

print(pred)
