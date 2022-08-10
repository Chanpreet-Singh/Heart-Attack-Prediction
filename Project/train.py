import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

import constants

class Train:
    def __init__(self):
        self.run = neptune.init(project=constants.project_id, api_token=constants.api_token)

    def read_data(self, file_path, train_size):
        df = pd.read_csv(file_path)
        y = df['target']
        X = df.drop(['target'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1-train_size, random_state=42)
        return X, y, X_train, X_test, y_train, y_test

    def main(self, params_dict, file_path, train_size):
        X, y, X_train, X_test, y_train, y_test = self.read_data(file_path, train_size)
        self.run["parameters"] = params_dict

        neptune_cbk = NeptuneCallback(run=self.run, base_namespace="training")
        nn_model = Sequential()
        nn_model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
        nn_model.add(Dense(10, activation='softmax'))
        # optimizer = tf.keras.optimizers.SGD(learning_rate=params_dict["lr"], momentum=params_dict["momentum"])
        optimizer = tf.keras.optimizers.Adam(lr=params_dict["lr"])
        nn_model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)
        nn_model.fit(X_train, y_train, epochs=params_dict["epochs"], batch_size=params_dict["batch_size"], callbacks=[neptune_cbk], validation_data=(X_test, y_test))
        nn_model.save(constants.model_output_file)

        eval_metrics = nn_model.evaluate(X_test, y_test, verbose=0)
        for j, metric in enumerate(eval_metrics):
            self.run["eval/{}".format(nn_model.metrics_names[j])] = metric

        self.run.stop()