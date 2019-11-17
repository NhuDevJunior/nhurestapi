from flask import Flask, render_template, request
from keras.preprocessing.image import img_to_array
import tensorflow as tf 
from PIL import Image
import numpy as np
import flask
import io
from flask import json,jsonify
# initialize our Flask application and the Keras model
app = flask.Flask(__name__)
model = None
def load_model():
    global model
    app.model = tf.keras.models.load_model('model2.h5')
    model=app.model

def prepare_image(image, target):

    if image.mode != "RGB":
        image = image.convert("RGB")


    image = image.resize(target)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    return image
@app.route('/',methods=["GET","POST"])
def hello():
    return "HELLO!"
@app.route("/predict", methods=["POST"])
def doan():
    if flask.request.method == "POST":
        if flask.request.files.get("image"):

            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image))

            image = prepare_image(image, target=(96, 96))

            preds = model.predict(image)
            idx = np.argmax(preds)
            ngoclinh1="0"
            hanquoc1="0"
            if(idx==0):
                ngoclinh1="1"
            else:
                hanquoc1="1"

    return jsonify(
           ngoclinh= ngoclinh1,
           hanquoc= hanquoc1
    )
if __name__ == "__main__":
    load_model()
    app.run(debug=True)