from flask import Flask, request, jsonify
import requests
import numpy as np
import os
from flask_cors import CORS
from tensorflow import keras
from tensorflow.keras.preprocessing import image

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


@ app.route('/', methods=['POST'])
def predict():
    if request.files['file']:
        image = request.files['file']
        probs = get_prediction(image=image)
        return jsonify(probs)


if __name__ == "__main__":
    app.run(debug=True)

 # test
resp = requests.post("http://localhost:5000",
                     files={"file": open(os.path.join(os.getcwd(), 'dataset', 'test', 'dogs', 'dog.0.jpg'), 'rb')})
print(resp.json())
