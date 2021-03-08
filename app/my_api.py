import numpy as np
import pandas as pd
import io
from PIL import Image

from tensorflow.keras.models import load_model  
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array

import flask
from flask import Flask, request
import flasgger
from flasgger import Swagger

threshold=0.2
app=Flask(__name__)
Swagger(app)

def load_mymodel():
    global model
    model = load_model('model.h5')

def prepare_image(image, target):
    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image=image/255
    return image

@app.route('/')
def welcome():
    return "Welcome All - go to http://127.0.0.1:5000/apidocs"

@app.route('/predict_file',methods=["POST"])
def predict_note_file():
    """Let's Predict if there is a crack in the pavement  
    This is using docstrings for specifications.
    ---
    parameters:
      - name: file
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    data={"success":False}
    
    if flask.request.method == "POST":
        image = flask.request.files["file"].read()
        image = Image.open(io.BytesIO(image))
        image = prepare_image(image, target=(256, 256))
        if model.predict(image)[0][0] >= threshold:
            data["predicted_class"] = "Cracked"
        else:
            data["predicted_class"] = "Non-Cracked"
        data["success"]=True
    
    return flask.jsonify(data)

if __name__=='__main__':
    #app.run(host='0.0.0.0',port=8000)
    load_mymodel()
    app.run(host="0.0.0.0")
		