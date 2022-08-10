import traceback
import pandas as pd
from numpy import argmax
from tensorflow import keras
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import constants
from train import Train

model = keras.models.load_model(constants.model_output_file)


def predict(x):
    message = "Error while predicting.."
    try:
        pred = model.predict(x)
        label = argmax(pred, axis=-1).astype('int')
        if label == 1:
            message = "Person is prone to heart attack"
        else:
            message = "Person is not prone to heart attack"
    except Exception as e:
        print("{0}\n{1}".format(e, traceback.format_exc()))
    return message

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route("/train", methods=["POST"])
def train():
    data = request.files['file']
    data.save(secure_filename(data.filename))
    params = {"epochs": int(request.form.get('epochs')),
              "batch_size": int(request.form.get('batch_size')),
              "lr": float(request.form.get('lr'))
              }
    Train().main(params_dict=params, file_path=data.filename, train_size=float(request.form.get('train_size')))
    return "Training Successful"
@app.route("/predict", methods=["POST"])
def test():
    data = request.form
    data = dict(data)
    data2 = data.copy()
    inputData = [float(value) for key, value in data2.items()]
    print(inputData)
    df = pd.DataFrame([inputData],
                      columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak",
                               "slope", "ca", "thal"])
    model_output = predict(df)
    print(model_output)
    if model_output == "Error while predicting..":
        return jsonify(model_output), 500
    else:
        return jsonify(model_output), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
