import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# we'll use tensorflow and keras for neural networks
import tensorflow as tf
import tensorflow.keras as keras

import utils  # we need this


def get_model(part: str, model_path: str = None):
    if part == "part1":
        target_model_fp = "./target-model.h5"
        model, _ = utils.load_model(target_model_fp)
        return model
    elif part == "part2":
        if model_path:
            model, _ = utils.load_model(model_path)
            return model
        # ConvNeXt models expect their inputs to be float or uint8 tensors of pixels with values in the [0-255] range.
        # setup model for CIFAR-10 classification
        model = keras.applications.ConvNeXtSmall(include_top=False, input_shape=(32, 32, 3), pooling="avg")
        # add in top layers for classification
        fc = keras.layers.Dense(512, activation="relu")(model.output)
        fc = keras.layers.Dense(256, activation="relu")(fc)
        fc = keras.layers.Dense(128, activation="relu")(fc)
        fc = keras.layers.Dense(10, activation="softmax")(fc)

        model = keras.Model(inputs=model.input, outputs=fc, name="part2_model")

        # compile model for training/testing
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model
    else:
        return None


def train_model(model, train_data: tuple, validation_data: tuple, test_data: tuple):
    batch_size = 32
    epochs = 10
    checkpointer = keras.callbacks.ModelCheckpoint(
        f"{model.name}_best.h5", verbose=1, save_best_only=True, monitor="val_loss"
    )

    history = model.fit(
        train_data[0],
        train_data[1],
        batch_size=batch_size,
        callbacks=[checkpointer],
        validation_data=validation_data,
        epochs=epochs,
        workers=4,
        use_multiprocessing=True,
    )

    print("Testing data results:")
    model.evaluate(test_data)

    return history


if __name__ == "__main__":
    part = "part2"
    model = get_model(part)

    if part == "part1":
        train_x, train_y, test_x, test_y, val_x, val_y, labels = utils.load_data()
    elif part == "part2":
        (train_x, train_y), (test_x, test_y) = keras.datasets.cifar10.load_data()
        # split testing data into validation and testing
        test_x, val_x, test_y, val_y = train_test_split(test_x, test_y, test_size=0.5)

    train_model(model, (train_x, train_y), (val_x, val_y), (test_x, test_y))
