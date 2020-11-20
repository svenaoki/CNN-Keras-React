from flask import Flask, request, jsonify
import requests
import numpy as np
import os
from PIL import Image
from flask_cors import CORS
from tensorflow import keras

app = Flask(__name__)
CORS(app)

PATH = os.path.join(os.getcwd(), 'backend')


def load_model():
    model = keras.models.load_model(os.path.join(PATH, 'entire_model.h5'))
    return model


def transform_image(input_image):
    img = Image.open(input_image)
    target_size = 128, 128
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def get_prediction(image):
    model = load_model()
    transformed_img = transform_image(image)
    predictions = model.predict(transformed_img)
    return np.around(predictions[0], decimals=2).tolist()


@app.route('/', methods=['POST'])
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
